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
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    orig_embd: int =1024

class GPT(nn.Module):
    def __init__(self, config, prev_config=None):
        super().__init__()
        self.config = config
        self.distillation_mode = False
        self.prev_max_depth = prev_config.n_layer if prev_config else None

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

    def forward(self, idx, targets=None, return_logits=True, gamma=0.2, max_depth=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for i, block in enumerate(self.transformer.h):
            if max_depth is not None and i >= max_depth:
                break
            x = block(x)
            if self.distillation_mode and self.prev_max_depth and i == self.prev_max_depth - 1:
                intermediate_logits = self.lm_head(x).detach()

        if self.distillation_mode and self.prev_max_depth:
            intermediate_logits = rmsnorm(intermediate_logits)

        x = rmsnorm(x)

        logits = self.lm_head(x)
        loss = None
        distill_loss = None

        if targets is not None:
            ground_truth_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            if self.distillation_mode and intermediate_logits is not None:
                target_output = logits.transpose(2, 1)
                targ = F.softmax(target_output, dim=1)
                distill_loss = F.cross_entropy(intermediate_logits.transpose(2, 1), targ, reduction='mean')
                loss = ground_truth_loss + gamma * distill_loss
            else:
                loss = ground_truth_loss
        else:
            logits = self.lm_head(x[:, [-1], :]) # inference-time mini-optimization

        if not return_logits:
            logits = None

        return logits, loss, ground_truth_loss, distill_loss
    

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        raise ValueError("ERROR: magic number mismatch in the data .bin file!")
    assert header[1] == 1, "unsupported version"
    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DataLoader:
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

def evaluate_model(pre_trained_model_path, val_data_pattern, batch_size=40, sequence_length=1024, val_max_steps=20, depth=None):
    # Initialize wandb
    wandb.init(project="gpt2_evaluation", config={
        "pre_trained_model_path": pre_trained_model_path,
        "val_data_pattern": val_data_pattern,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "val_max_steps": val_max_steps,
        "depth": depth,
    })
    
    # Load the pre-trained model's state dictionary
    checkpoint = torch.load(pre_trained_model_path)
    state_dict = checkpoint['model_state_dict']

    # Load the pre-trained model
    model_config = GPTConfig(vocab_size=50257, n_layer=24, n_head=16, n_embd=1024)
    model = GPT(model_config)

    # Remove the "_orig_mod." prefix from the keys in the state_dict
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()

    # Load the validation data
    val_loader = DataLoader(val_data_pattern, batch_size, sequence_length)

    # Evaluate the model
    val_loader.reset()
    with torch.no_grad():
        val_loss = 0.0
        for _ in range(val_max_steps):
            x_val, y_val = val_loader.next_batch()
            _, loss, _, _ = model(x_val, y_val, return_logits=False, max_depth=depth)
            val_loss += loss.item()
        val_loss /= val_max_steps

    print(f"Validation loss (depth={depth}): {val_loss}")
    wandb.log({"validation_loss": val_loss, "depth": depth})


if __name__ == "__main__":
    pre_trained_model_path = "/home/bsnaas/git/gpt-distill/model/baseline.pt"
    val_data_pattern = "data/fineweb10B/fineweb_val_*.bin"
    evaluate_model(pre_trained_model_path, val_data_pattern, depth=12)
