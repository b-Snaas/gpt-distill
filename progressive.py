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

from torch.profiler import profile, ProfilerActivity, record_function

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
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))

        # Add student embedding and lm_head
        self.student_wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.student_lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.student_lm_head.LLMC_SKIP_INIT = 1
        self.student_wte.weight = self.student_lm_head.weight

        self.apply(self._init_weights)

        # Flag to determine which embedding/lm_head pair to use
        self.use_student = True

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        # Use student or teacher embedding based on the flag
        tok_emb = self.student_wte(idx) if self.use_student else self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # Use student or teacher lm_head based on the flag
            logits = self.student_lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.student_lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Always optimize student parameters and shared transformer blocks
        params_to_optimize = (
            list(self.student_wte.parameters()) +
            list(self.student_lm_head.parameters()) +
            list(self.transformer.h.parameters()) +
            list(self.transformer.wpe.parameters())
        )
        
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=weight_decay, betas=betas)
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

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    print(*args, **kwargs)

def train(input_bin="data/fineweb10B/fineweb_train_*.bin", 
            input_val_bin="data/fineweb10B/fineweb_val_*.bin", 
            model_path=None, 
            batch_size=64, 
            sequence_length=512, 
            num_iterations=12288, 
            learning_rate=0.0018, 
            warmup_iters=256, 
            weight_decay=0.1,
            val_loss_every=128, 
            val_max_steps=20,
            ):

    # Initialize wandb
    wandb.init(project="gpt2_distill", config={
        "input_bin": input_bin,
        "input_val_bin": input_val_bin,
        "output_dir": model_path,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "warmup_iters": warmup_iters,
        "weight_decay": weight_decay,
        "val_loss_every": val_loss_every,
        "val_max_steps": val_max_steps,
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

    # load tokens
    train_loader = DataLoader(input_bin, B, T)
    val_loader = None
    if input_val_bin:
        val_loader = DataLoader(input_val_bin, B, T)
    x, y = train_loader.next_batch()

    def initialize_model(depth, prev_model=None):
        model_config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=depth, n_head=16, n_embd=1024)
        model = GPT(model_config)
        model = model.train().cuda()
        
        if prev_model is not None:
            with torch.no_grad():
                # Copy transformer blocks
                for i in range(min(len(prev_model.transformer.h), len(model.transformer.h))):
                    model.transformer.h[i].load_state_dict(prev_model.transformer.h[i].state_dict())
                
                # Copy position embeddings
                model.transformer.wpe.weight.copy_(prev_model.transformer.wpe.weight)
                # Copy student weights to new model's student weights
                model.student_wte.weight.copy_(prev_model.student_wte.weight)
                model.student_lm_head.weight.copy_(prev_model.student_lm_head.weight)
        
        return model

    def reinitialize_optimizer(model, learning_rate, weight_decay):
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
        return optimizer

    # progressive training schedule
    progressive_schedule = [(6, 5000), (24, 5000)]

    # Calculate total iterations in the progressive schedule
    total_scheduled_iters = sum(iters for _, iters in progressive_schedule)

    # Adjust the schedule if num_iterations is larger
    if num_iterations > total_scheduled_iters:
        last_depth = progressive_schedule[-1][0]
        extra_iters = num_iterations - total_scheduled_iters
        progressive_schedule.append((last_depth, extra_iters))


    # initialize the first model and optimizer
    current_depth, current_iters = progressive_schedule.pop(0)
    model = initialize_model(current_depth)
    optimizer = reinitialize_optimizer(model, learning_rate, weight_decay)

    raw_model = model

    # learning rate decay scheduler
    def get_lr(it, total_iters):
        assert it <= total_iters
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / warmup_iters
        # 2) linear decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        return (0.1 + (1 - decay_ratio)) / (0.1 + 1) * learning_rate

    timings = []
    instances_seen = 0  # Initialize counter for instances seen
    step = 0

    while step < num_iterations:
        if step >= current_iters:
            if progressive_schedule:
                # Move to the next depth stage
                current_depth, new_iters = progressive_schedule.pop(0)
                current_iters += new_iters

                # Free up the memory used by the old model and optimizer
                prev_model = model
                del optimizer
                torch.cuda.empty_cache()

                # Initialize the new model with weights from the previous model
                model = initialize_model(current_depth, prev_model)
                optimizer = reinitialize_optimizer(model, learning_rate, weight_decay)
                
                # Delete the previous model to free memory
                del prev_model

        t0 = time.time()

        # once in a while evaluate the validation dataset
        if (val_loss_every > 0 and (step % val_loss_every == 0 or step == num_iterations)) and (val_loader is not None):
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
            print0(f"val loss {val_loss}")
            wandb.log({"val_loss": val_loss, "step": step})

        if step == num_iterations:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        
        # forward pass
        forward_start = time.time()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        forward_end = time.time()
        forward_time = forward_end - forward_start

        start_batch = time.time() 
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # Increment the counter for instances seen
        instances_seen += x.size(0)
        end_batch = time.time()
        batch_time = end_batch - start_batch

        # backward pass
        backward_start = time.time()
        loss.backward()

        # Gradient normalization
        for p in model.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-6)
        # determine and set the learning rate for this iteration
        lr = get_lr(step, num_iterations)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        backward_end = time.time()
        backward_time = backward_end - backward_start

        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        start_cuda_sync = time.time()
        torch.cuda.synchronize()
        end_cuda_sync = time.time()
        cuda_sync_time = end_cuda_sync - start_cuda_sync
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = B * T / (t1-t0)
        lossf = loss.item() # keep track of the mean loss
        # print0(f"step {step+1:4d}/{num_iterations} | train loss {lossf:.6f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        wandb.log({
            "train_loss": lossf, 
            "step": step, 
            "instances_seen": instances_seen,
            "batch_time": t1 - t0,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "cuda_sync_time": cuda_sync_time,
            "batch_process_time": batch_time
        })  # Log training loss, instances seen, and timings to wandb

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > num_iterations - 20:
            timings.append(t1-t0)

        step += 1

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

if __name__ == "__main__":
    fire.Fire(train)