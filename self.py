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
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # Additional layers for distillation outputs
        self.distill_layers = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(4)])

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialize the position embedding at std=0.02 to match the scale of the token embedding.
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, current_depth, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape (t)

        # Forward the GPT model
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        # Calculate the distillation index based on the current depth
        # Explicitly convert to integer and handle potential division by zero
        num_blocks = len(self.transformer.h)

        quarter_depth = len(self.transformer.h) // 4
        dist_points = [quarter_depth - 1, (quarter_depth * 2) - 1, (quarter_depth * 3) - 1 , (quarter_depth * 4) - 1]

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


        # Apply the distillation layer for the current depth
        # x = rmsnorm(x)

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

        # Determine the deepest available distillation output
        if dist_output_4th is not None:
            y = self.distill_layers[3](dist_output_4th)
        elif dist_output_3rd is not None:
            y = self.distill_layers[2](dist_output_3rd)
        elif dist_output_2nd is not None:
            y = self.distill_layers[1](dist_output_2nd)
        elif dist_output_1st is not None:
            y = self.distill_layers[0](dist_output_1st)
        else:
            y = None

        y = rmsnorm(y)
        logits = y

        # Calculate the loss for the current depth if targets are provided
        if targets is not None and logits is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        # If not returning logits, set logits to None
        if not return_logits:
            logits = None

        return logits, loss
        

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer

    def print_network_depth(self):
        depth = len(self.transformer.h)
        print(f"Network depth (number of layers): {depth}")
        return depth

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

    def next_batch(self):
        B = self.B
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

    def update_batch_size(self, new_batch_size):
        self.B = new_batch_size

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    print(*args, **kwargs)

def save_model(model, path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

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

def train(input_bin="data/fineweb10B/fineweb_train_*.bin", 
          input_val_bin="data/fineweb10B/fineweb_val_*.bin", 
          batch_size=64, 
          sequence_length=512, 
          num_iterations=12288, 
          learning_rate=0.0018, 
          warmup_iters=256, 
          weight_decay=0.1,
          val_loss_every=128, 
          val_max_steps=20,
          depth=12,
          embedding=768,
          ):

    # Initialize wandb
    wandb.init(project="gpt2_distill", config={
        "input_bin": input_bin,
        "input_val_bin": input_val_bin,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "warmup_iters": warmup_iters,
        "weight_decay": weight_decay,
        "val_loss_every": val_loss_every,
        "val_max_steps": val_max_steps,
        "depth": depth,
        "embedding": embedding,
    })

    # args error checking and convenience variables
    B, T = batch_size, sequence_length

    assert 1 <= T <= 1024

    assert torch.cuda.is_available(), "CUDA is required"
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"using device: {device}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the model from scratch
    model_config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=depth, n_head=8, n_embd=embedding)
    model = GPT(model_config)
    model = model.train().cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    print0("compiling the model...")

    model = torch.compile(model)

    # load tokens
    train_loader = DataLoader(input_bin, B, T)
    val_loader = None
    if input_val_bin:
        val_loader = DataLoader(input_val_bin, B, T)
    x, y = train_loader.next_batch()

    raw_model = model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=weight_decay,
                                               learning_rate=learning_rate, betas=(0.9, 0.95),
                                               device_type=device)

    # learning rate decay scheduler
    def get_lr(it):
        assert it <= num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it+1) / warmup_iters
        # 2) linear decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (num_iterations - warmup_iters)
        assert 0 <= decay_ratio <= 1
        return (0.1 + (1 - decay_ratio)) / (0.1 + 1) * learning_rate

    run_id = str(uuid.uuid4())

    timings = []
    instances_seen = 0  # Initialize counter for instances seen

    quarter_depth = depth // 4
    batch_size_by_depth = {
        quarter_depth: batch_size + 30,
        2 * quarter_depth: batch_size + 20,
        3 * quarter_depth: batch_size + 10,
        depth: batch_size
    }

    lr_by_depth = {
        quarter_depth: 0.00018,
        2 * quarter_depth: 0.00018,
        3 * quarter_depth: 0.00018,
        depth: 0.00018
    }

    for step in range(num_iterations + 1):
        t0 = time.time()
        last_step = (step == num_iterations)

        # once in a while evaluate the validation dataset
        if (val_loss_every > 0 \
            and (step % val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    current_depth = get_layer_depth(step, num_iterations, depth)
                    _, loss = model(x_val, current_depth, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # Get the current depth for the forward pass
        current_depth = get_layer_depth(step, num_iterations, depth)
        B = batch_size_by_depth[current_depth]
        learning_rate = lr_by_depth[current_depth]

        # Update batch size of train_loader
        train_loader.update_batch_size(B)

        # forward pass
        with ctx:
            _, loss = model(x, current_depth, y, return_logits=False)
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # Increment the counter for instances seen
        instances_seen += x.size(0)
        # backward pass
        loss.backward()
        for layer in model.transformer.h[:current_depth]:
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad = p.grad / (p.grad.norm() + 1e-6)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = B * T / (t1-t0)
        lossf = loss.item() # keep track of the mean loss
        # print0(f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        wandb.log({"train_loss": lossf, "step": step, "instances_seen": instances_seen})  # Log training loss and instances seen to wandb

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

if __name__ == "__main__":
    fire.Fire(train)
