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
    orig_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config, prev_config=None):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.orig_embd),
            h = nn.ModuleList()
        ))

        # Initialize layers
        for i in range(config.n_layer):
            if prev_config and i < prev_config.n_layer:
                # Use previous configuration for copied layers
                self.transformer.h.append(Block(prev_config))
            else:
                # Use new configuration for new layers
                self.transformer.h.append(Block(config))

        self.lm_head = nn.Linear(config.orig_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        if config.n_embd != config.orig_embd:
            self.transformer.proj_down = nn.Linear(config.orig_embd, config.n_embd)
            self.transformer.proj_up = nn.Linear(config.n_embd, config.orig_embd)


    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for i, block in enumerate(self.transformer.h):
            x = block(x)
            if i == 11 and self.config.n_embd != self.config.orig_embd:
                x = self.transformer.proj_down(x)  # Project back up after 12th layer

        if self.config.n_embd != self.config.orig_embd:
            x = self.transformer.proj_up(x)  # Final projection up

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

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    print(*args, **kwargs)

def print_model_details(model):
    print(f"GPT Model Structure:")
    print(f"  Embedding layer: input size = {model.transformer.wte.num_embeddings}, output size = {model.transformer.wte.embedding_dim}")
    print(f"  Number of transformer layers: {len(model.transformer.h)}")
    print(f"  Language model head: input size = {model.lm_head.in_features}, output size = {model.lm_head.out_features}")
    
    if hasattr(model.transformer, 'proj'):
        print(f"  Projection layer: input size = {model.transformer.proj.in_features}, output size = {model.transformer.proj.out_features}")
    
    for i, layer in enumerate(model.transformer.h):
        print(f"  Transformer layer {i + 1}:")
        print(f"    Input dimension: {layer.mlp.c_fc.in_features}")
        print(f"    Output dimension: {layer.mlp.c_proj.out_features}")
        print(f"    Number of attention heads: {layer.attn.n_head}")
    

def train(input_bin="data/fineweb10B/fineweb_train_*.bin", 
            input_val_bin="data/fineweb10B/fineweb_val_*.bin", 
            model_path=None,  
            sequence_length=1024,
            warmup_iters=250,
            weight_decay=0.1,
            val_loss_every=1280, 
            val_max_steps=20,
            ):

    # Initialize wandb
    wandb.init(project="gpt2_distill", config={
        "input_bin": input_bin,
        "input_val_bin": input_val_bin,
        "output_dir": model_path,
        "sequence_length": sequence_length,
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

    def initialize_model(depth, prev_model=None, n_head=12, n_embd=768):
        if prev_model:
            orig_embd = prev_model.config.orig_embd
            prev_config = prev_model.config
        else:
            orig_embd = n_embd
            prev_config = None

        model_config = GPTConfig(vocab_size=50257, n_layer=depth, n_head=n_head, n_embd=n_embd, orig_embd=orig_embd)
        model = GPT(model_config, prev_config)
        model = model.train().cuda()

        if prev_model:
            # Copy weights for the first 12 layers
            for i in range(min(12, len(prev_model.transformer.h))):
                model.transformer.h[i].load_state_dict(prev_model.transformer.h[i].state_dict())
            
            # Copy embedding weights
            model.transformer.wte.weight.data.copy_(prev_model.transformer.wte.weight.data)

        model = torch.compile(model)
        return model

    def reinitialize_optimizer(model, learning_rate, weight_decay):
        optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(0.9, 0.95), device_type=device)
        return optimizer
    
    # learning rate decay scheduler (linear warmup and final warmdown)
    def get_lr(it, stage_start_iter, current_iters, warmdown_iters, current_lr):
        stage_progress = it - stage_start_iter
        assert stage_progress <= current_iters
        # 1) linear warmup for warmup_iters steps
        if stage_progress < warmup_iters:
            # print(f"WARMING UP iteration:{it} and stage_start: {stage_start_iter} and current_iters: {current_iters}")
            return current_lr * (stage_progress + 1) / warmup_iters
        # 2) constant lr for most of the stage
        elif stage_progress < current_iters - warmdown_iters:
            # print(f"NORMAL iteration:{it} and stage_start: {stage_start_iter} and current_iters: {current_iters}")
            return current_lr
        # 3) linear warmdown
        else:
            # print(f"WARMING DOWN iteration:{it} and stage_start: {stage_start_iter} and current_iters: {current_iters}")
            decay_ratio = (current_iters - stage_progress) / warmdown_iters
            return current_lr * decay_ratio

    # progressive training schedule
    progressive_schedule = [
        (12, 12, 768, 1000, 1, 0.00018),
        (24, 9, 576, 1000, 1, 0.0001)
    ]

    # Calculate total iterations in the progressive schedule
    total_scheduled_iters = sum(iters for _, _, _, iters, _, _ in progressive_schedule)

    # initialize the first model and optimizer
    current_iters = 0
    current_depth, current_head, current_embd, new_iters, current_batch_size, current_lr = progressive_schedule.pop(0)
    current_iters += new_iters
    warmdown_iters = int(0.2 * current_iters)
    stage_start_iter = 0
    model = initialize_model(current_depth, n_head=current_head, n_embd=current_embd)
    optimizer = reinitialize_optimizer(model, current_lr, weight_decay)

    print_model_details(model)

    # load tokens
    train_loader = DataLoader(input_bin, current_batch_size, sequence_length)
    val_loader = None
    if input_val_bin:
        val_loader = DataLoader(input_val_bin, current_batch_size, sequence_length)
    x, y = train_loader.next_batch()

    timings = []
    instances_seen = 0
    step = 0
    steps_in_current_schedule = 0
    steps_in_prev_schedules = 0
    local_step = 0

    num_iterations = total_scheduled_iters

    while step < num_iterations:
        if progressive_schedule and (steps_in_current_schedule + steps_in_prev_schedules) == current_iters - 10:
            next_depth, _, _ , _ , next_batch_size, new_lr = progressive_schedule[0]
            if train_loader.B != next_batch_size:
                print(f"Switching to next batch size {next_batch_size} at step {step}")
                train_loader.set_batch_size(next_batch_size)
                val_loader.set_batch_size(next_batch_size)
                current_lr = new_lr
        if step >= current_iters:
            if progressive_schedule:
                prev_model = model
                current_depth, current_head, current_embd, new_iters, new_batch_size, new_lr = progressive_schedule.pop(0)
                current_iters += new_iters
                warmdown_iters = int(0.2 * new_iters)
                steps_in_prev_schedules += steps_in_current_schedule
                steps_in_current_schedule = 0
                local_step = 0

                del optimizer

                clear_memory()

                # Initialize the new model with weights from the previous model
                model = initialize_model(current_depth, prev_model=prev_model, n_head=current_head, n_embd=current_embd)
                optimizer = reinitialize_optimizer(model, new_lr, weight_decay)
                
                # Set the batch size for the new stage
                train_loader.set_batch_size(new_batch_size)
                val_loader.set_batch_size(new_batch_size)

                # Delete the previous model to free memory
                del prev_model

                stage_start_iter = step
                current_lr = new_lr  # Update the current learning rate

                clear_memory()

                print_model_details(model)

        t0 = time.time()

        # once in a while evaluate the validation dataset
        if (val_loss_every > 0 and (step % val_loss_every == 0 or step == num_iterations)) and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                val_ground = 0.0
                for _ in range(val_max_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model(
                        x_val, y_val, 
                        return_logits=False, 
                    )
                    val_loss += loss.item()
                val_loss /= val_max_steps

            wandb.log({"val_loss": val_loss, "step": step})

        if step == num_iterations:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        
        with ctx:
            _, loss = model(
                x, y, 
                return_logits=False, 
            )
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # Increment the counter for instances seen
        instances_seen += x.size(0)

        loss.backward()

        # Gradient normalization
        for p in model.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-6)

        lr = get_lr(step, stage_start_iter, new_iters, warmdown_iters, current_lr)
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
        lossf = loss.item() # keep track of the mean loss

        log_dict = {
            "train_loss": lossf, 
            "step": step, 
            "instances_seen": instances_seen,
            "lr": lr,
        }

        wandb.log(log_dict)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > num_iterations - 20:
            timings.append(t1-t0)

        step += 1
        steps_in_current_schedule += 1
        local_step += 1  # Increment local step

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

if __name__ == "__main__":
    fire.Fire(train)