import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass
import time
import gc
import wandb
import fire
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
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
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer
    
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
        self.filename_pattern = filename_pattern
        self.T = T
        self.set_batch_size(B)
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

    def set_batch_size(self, B):
        self.B = B

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
    
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
    # Force GPU to release memory
    torch.cuda.synchronize()


def train(input_bin="data/fineweb10B/fineweb_train_*.bin", 
            input_val_bin="data/fineweb10B/fineweb_val_*.bin", 
            model_path=None, 
            model="d12", 
            sequence_length=1024, 
            num_iterations=200000, 
            learning_rate=0.00018, 
            warmup_iters=500,
            warmdown_iters=20000,
            weight_decay=0.1,
            val_loss_every=1280, 
            val_max_steps=20,
            ):

    # Initialize wandb
    wandb.init(project="gpt2_distill", config={
        "input_bin": input_bin,
        "input_val_bin": input_val_bin,
        "output_dir": model_path,
        "model": model,
        "sequence_length": sequence_length,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "warmup_iters": warmup_iters,
        "weight_decay": weight_decay,
        "val_loss_every": val_loss_every,
        "val_max_steps": val_max_steps,
    })


    assert torch.cuda.is_available(), "CUDA is required"
    device = 'cuda:0'
    torch.cuda.set_device(device)
    print(f"using device: {device}")

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    def initialize_model(depth, prev_model=None):
        model_config = GPTConfig(vocab_size=50257, n_layer=depth, n_head=25, n_embd=1600)
        model = GPT(model_config)
        model = model.train().cuda()
        
        if prev_model is not None:
            # Copy transformer blocks
            for i in range(min(len(prev_model.transformer.h), len(model.transformer.h))):
                model.transformer.h[i].load_state_dict(prev_model.transformer.h[i].state_dict())

            # copy embedding and lm_head
            model.transformer.wte.load_state_dict(prev_model.transformer.wte.state_dict())
            model.lm_head.load_state_dict(prev_model.lm_head.state_dict())
        
        compile_start_time = time.time()
        model = torch.compile(model)
        compile_end_time = time.time()
        compile_duration = compile_end_time - compile_start_time
        print(f"Model compilation for depth {depth} took {compile_duration:.2f} seconds")
        
        return model

    def reinitialize_optimizer(model, learning_rate, weight_decay):
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
        return optimizer
    
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee

    # learning rate decay scheduler (linear warmup and final warmdown)
    def get_lr(it, stage_start_iter, current_stage_iters, is_final_stage):
        stage_progress = it - stage_start_iter
        assert stage_progress <= current_stage_iters
        # 1) linear warmup for warmup_iters steps
        if stage_progress < warmup_iters:
            return learning_rate * (stage_progress + 1) / warmup_iters
        # 2) constant lr for most of the stage
        elif not is_final_stage or stage_progress < current_stage_iters - warmdown_iters:
            return learning_rate
        # 3) linear warmdown (only in the final stage)
        else:
            decay_ratio = (current_stage_iters - stage_progress) / warmdown_iters
            return learning_rate * decay_ratio

    timings = []
    instances_seen = 0  # Initialize counter for instances seen

    # progressive training schedule
    progressive_schedule = [
        (12, 10000, 45, 0.00020),
        (48, 190000, 10, 0.00005),
        (3, 10000, 75, 0.00025),
        (6, 10000, 70, 0.00025),
    ]

    # initialize the first model and optimizer
    current_depth, current_stage_iters, current_batch_size, current_lr = progressive_schedule.pop(0)
    stage_start_iter = 0
    model = initialize_model(current_depth)
    optimizer = reinitialize_optimizer(model, current_lr, weight_decay)

    # load tokens
    train_loader = DataLoader(input_bin, current_batch_size, sequence_length)
    val_loader = None
    if input_val_bin:
        val_loader = DataLoader(input_val_bin, current_batch_size, sequence_length)
    x, y = train_loader.next_batch()

    instances_seen = 0  # Initialize counter for instances seen
    step = 0

    for step in range(num_iterations + 1):
        if step >= stage_start_iter + current_stage_iters:
            if progressive_schedule:
                # Move to the next depth stage
                current_depth, new_stage_iters, new_batch_size, new_lr = progressive_schedule.pop(0)
                stage_start_iter += current_stage_iters
                current_stage_iters = new_stage_iters

                # Free up the memory used by the old model and optimizer
                prev_model = model
                del optimizer

                clear_memory()

                # Initialize the new model with weights from the previous model
                model = initialize_model(current_depth, prev_model)

                optimizer = reinitialize_optimizer(model, new_lr, weight_decay)
                
                # Set the batch size for the new stage
                train_loader.set_batch_size(new_batch_size)
                val_loader.set_batch_size(new_batch_size)

                # Delete the previous model to free memory
                del prev_model

                current_lr = new_lr  # Update the current learning rate

                clear_memory()

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
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.item()
                val_loss /= val_max_steps
            # log to console and to file
            print(f"val loss {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        
        with ctx:
            _, loss = model(x, y, return_logits=False)

        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # Increment the counter for instances seen
        instances_seen += x.size(0)
        loss.backward()

        for p in model.parameters():
            p.grad = p.grad / (p.grad.norm() + 1e-6)
        # determine and set the learning rate for this iteration
        is_final_stage = (len(progressive_schedule) == 0)
        lr = get_lr(step, stage_start_iter, current_stage_iters, is_final_stage)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()

        lossf = loss.item() # keep track of the mean loss
        # print0(f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        wandb.log({
            "train_loss": lossf, 
            "step": step, 
            "instances_seen": instances_seen,
        })  # Log training loss, instances seen, and timings to wandb

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]

if __name__ == "__main__":
    fire.Fire(train)
