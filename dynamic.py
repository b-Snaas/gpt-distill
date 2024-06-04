import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config

import fire
import wandb

torch.set_float32_matmul_precision('high')

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        x = x + self.attn_scale * self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config, shared_dist_layer_weights=True):
        super().__init__()
        self.config = config
        self.shared_dist_layer_weights = shared_dist_layer_weights

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.dist_layers = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(4)])
        
        if self.shared_dist_layer_weights:
            for layer in self.dist_layers:
                layer.weight = self.transformer.wte.weight
                layer.LLMC_SKIP_INIT = 1
        else:
            for layer in self.dist_layers:
                layer.LLMC_SKIP_INIT = 1
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True, current_depth=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        fourth_depth = len(self.transformer.h) // 4
        dist_points = [fourth_depth - 1, 2 * fourth_depth - 1, 3 * fourth_depth - 1, len(self.transformer.h) - 1]

        dist_output_1st = None
        dist_output_2nd = None
        dist_output_3rd = None
        dist_output_4th = None

        for i, block in enumerate(self.transformer.h[:current_depth]):
            x = block(x)

            if i == dist_points[0]:
                dist_output_1st = x
            elif i == dist_points[1]:
                dist_output_2nd = x
            elif i == dist_points[2]:
                dist_output_3rd = x
            elif i == dist_points[3]:
                dist_output_4th = x

        x = rmsnorm(x)

        y_1st = None if dist_output_1st is None else self.dist_layers[0](dist_output_1st)
        y_2nd = None if dist_output_2nd is None else self.dist_layers[1](dist_output_2nd)
        y_3rd = None if dist_output_3rd is None else self.dist_layers[2](dist_output_3rd)
        y_4th = None if dist_output_4th is None else self.dist_layers[3](dist_output_4th)

        losses = []
        for i, y in enumerate([y_1st, y_2nd, y_3rd, y_4th]):
            if y is not None and i < current_depth // fourth_depth:
                loss = F.cross_entropy(y.view(-1, y.size(-1)), targets.view(-1), ignore_index=-1)
                losses.append(loss)
            else:
                losses.append(None)

        if not return_logits:
            y_1st, y_2nd, y_3rd, y_4th = None, None, None, None

        return losses, y_1st, y_2nd, y_3rd, y_4th

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer

# -----------------------------------------------------------------------------
# Our own simple Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DataLoader:
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self, batch_size=None):
        B = batch_size if batch_size is not None else self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

def get_layer_depth(batch_num, num_batches, depth):
    quarter_depth = depth // 4
    phases = [0.25, 0.5, 0.75, 1.0]
    weights = [
        [0.6, 0.2, 0.1, 0.1],
        [0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.1, 0.2, 0.7]  
    ]
    # Safeguard against the ratio exactly equalling 1.0
    phase_index = next((i for i, phase in enumerate(phases) if batch_num / num_batches <= phase), len(phases) - 1)
    current_weights = weights[phase_index]
    chosen_depth = np.random.choice(
        [quarter_depth, 2 * quarter_depth, 3 * quarter_depth, depth],
        p=current_weights
    )
    return chosen_depth

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    print(*args, **kwargs)

def log_memory_usage(event_name, current_depth=None, batch_size=None):
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2) # Convert bytes to MB
    max_allocated_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert bytes to MB
    print0(f"{event_name}: Allocated Memory: {allocated_memory:.2f} MB, Max Allocated Memory: {max_allocated_memory:.2f} MB, Depth: {current_depth}, Batch Size: {batch_size}")

def train(input_bin="data/fineweb10B/fineweb_train_*.bin", 
          input_val_bin="data/fineweb10B/fineweb_val_*.bin", 
          output_dir=None, 
          model="d12", 
          sequence_length=1024, 
          num_iterations=12288, 
          warmup_iters=256, 
          weight_decay=0.1, 
          val_loss_every=128, 
          val_max_steps=20):
    
    wandb.init(project="gpt2_distill", config={
        "input_bin": input_bin,
        "input_val_bin": input_val_bin,
        "model": model,
        "sequence_length": sequence_length,
        "num_iterations": num_iterations,
        "warmup_iters": warmup_iters,
        "weight_decay": weight_decay,
        "val_loss_every": val_loss_every,
        "val_max_steps": val_max_steps
    })

    print0(f"Running pytorch {torch.version.__version__}")

    T = sequence_length
    assert 1 <= T <= 1024
    assert model in {"d12", "d24", "d36", "d48"}

    assert torch.cuda.is_available(), "CUDA is required"
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"using device: {device}")

    log_memory_usage("Start of Training")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the model from scratch
    model_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[model]
    model = GPT(model_config, shared_dist_layer_weights=True)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(model)

    # Define depth and batch size mappings
    depth = model_config.n_layer
    batch_size_by_depth = {
        depth // 4: 60,
        2 * (depth // 4): 40,
        3 * (depth // 4): 28,
        depth: 21
    }

    lr_by_depth = {
        depth // 4: 0.0005,
        2 * (depth // 4): 0.00035,
        3 * (depth // 4): 0.00025,
        depth: 0.00018
    }

    # Initial dynamic depth selection
    current_depth = get_layer_depth(0, num_iterations, depth)
    B = batch_size_by_depth[current_depth]
    base_lr = lr_by_depth[current_depth]

    # load tokens
    train_loader = DataLoader(input_bin, B, T)
    val_loader = None
    if input_val_bin:
        val_loader = DataLoader(input_val_bin, B, T)
    x, y = train_loader.next_batch(batch_size=B)

    raw_model = model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay,
                                               learning_rate=base_lr, betas=(0.9, 0.95),
                                               device_type=device)

    # learning rate decay scheduler with warmup
    def get_lr(it, base_lr):
        assert it <= num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return base_lr * (it + 1) / warmup_iters
        # 2) linear decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
        assert 0 <= decay_ratio <= 1
        return (0.1 + (1 - decay_ratio)) / (0.1 + 1) * base_lr

    run_id = str(uuid.uuid4())

    timings = []

    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = (step == num_iterations)

        if (val_loss_every > 0 \
            and (step % val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_losses = [0.0] * 4
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    losses, _, _, _, _ = model(x_val, y_val, return_logits=False, current_depth=depth)
                    for i, loss in enumerate(losses):
                        if loss is not None:
                            val_losses[i] += loss.item()
                val_losses = [loss / val_max_steps for loss in val_losses]
            # log to console and to wandb
            print0(f"val losses {val_losses}")
            val_loss_dict = {f"val_loss_{i+1}": loss for i, loss in enumerate(val_losses)}
            wandb.log(val_loss_dict, step=step)

        if last_step:
            break

        # Dynamically determine depth
        current_depth = get_layer_depth(step, num_iterations, depth)
        B = batch_size_by_depth[current_depth]
        base_lr = lr_by_depth[current_depth]

        # Determine learning rate with warmup and decay
        lr = get_lr(step, base_lr)

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # forward pass
        log_memory_usage(f"Before Forward Pass, Step {step+1}", current_depth, B)
        with ctx:
            losses, y_1st, y_2nd, y_3rd, y_4th = model(x, y, return_logits=False, current_depth=current_depth)
        log_memory_usage(f"After Forward Pass, Step {step+1}", current_depth, B)

        # advance the dataset for the next batch
        x, y = train_loader.next_batch(batch_size=B)

        # backward pass
        current_loss = losses[-1]
        current_loss.backward()
        log_memory_usage(f"After Backward Pass, Step {step+1}", current_depth, B)

        for p in model.parameters():
            p.grad = p.grad / (p.grad.norm() + 1e-6)

        # set the learning rate for this iteration
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log_memory_usage(f"After Optimizer Step, Step {step+1}", current_depth, B)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = B * T / (t1 - t0)
        lossf = current_loss.item() # keep track of the mean loss
        print0(f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | ({(t1 - t0) * 1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to wandb
        loss_dict = {f"train_loss_{i+1}": loss.item() if loss is not None else None for i, loss in enumerate(losses)}
        wandb.log(loss_dict, step=step)
        wandb.log({"output_layer_loss": lossf, "lr": lr, "tokens_per_second": tokens_per_second}, step=step)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > num_iterations - 20:
            timings.append(t1 - t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings) * 1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------

    log = dict(model=raw_model.state_dict(), code=code, args=locals())
    os.makedirs('logs', exist_ok=True)
    torch.save(log, 'logs/%s.pt' % run_id)

if __name__ == "__main__":
    fire.Fire(train)
