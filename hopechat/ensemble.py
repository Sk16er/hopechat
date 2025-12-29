"""
HOPE: Hierarchical Optimizing Processing Ensemble.
A self-modifying, multi-scale learning architecture.
"""

import math
from functools import partial
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hopechat.common import get_dist_info, print0
from hopechat.muon import Muon, DistMuon
from hopechat.adamw import DistAdamW

@dataclass
class HOPEConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12 # Initial nesting levels
    max_layers: int = 20 # Maximum possible levels (for dynamic growth)
    n_head: int = 6 
    n_kv_head: int = 6 
    n_embd: int = 768
    chunk_sizes: List[int] = field(default_factory=lambda: [1, 4, 16, 64, 256]) # CMS chunk sizes per level

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class MetaController(nn.Module):
    """
    Monitors a composite meta-loss and controls the dynamic hierarchy.
    """
    def __init__(self, config):
        super().__init__()
        self.threshold = 0.5 # Delta threshold
        self.current_levels = config.n_layer
        self.max_levels = config.max_layers
        # Meta-parameters (learnable thresholds or weights for the decision)
        self.growth_gate = nn.Parameter(torch.tensor(0.0)) 

    def forward(self, current_loss):
        return self.current_levels

    def check_growth(self, loss_val):
        # Called at the end of an epoch or step to potentially increase levels
        if loss_val > self.threshold and self.current_levels < self.max_levels:
            self.current_levels += 1
            return True
        return False

class NestedAdaptationBlock(nn.Module):
    """
    Nested Learning Block (formerly Attention).
    Includes Surprise-Based Memory Update and CMS.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # Determine Chunk Size for this level (CMS)
        level_idx = layer_idx % len(config.chunk_sizes)
        self.chunk_size = config.chunk_sizes[level_idx]
        
        # Level 2 (Slow Weights)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        
        # Meta-Network for Surprise Gate
        # Generates the "Surprise Gate" based on input x.
        self.meta_net = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd // 4),
            nn.ReLU(),
            nn.Linear(self.n_embd // 4, self.n_head)
        )
        
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x, state_cache=None):
        B, T, C = x.size()
        
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        
        # Surprise Gate: G_t = Sigmoid(MetaNet(x_t))
        # This modulates the update speed. High surprise -> Fast update. Low surprise -> Stable.
        surprise = torch.sigmoid(self.meta_net(x)).view(B, T, self.n_head, 1)

        # CMS: We process in chunks.
        # Update Rule: S_t = (1 - G_t) * S_{t-1} + G_t * (K^T V)
        # This is exactly Linear Nested Optimization with data-dependent decay.
        
        decay = 1.0 - surprise

        if state_cache is not None and state_cache.is_inference:
            # Recurrent Mode
            prev_state = state_cache.get_state(self.layer_idx)
            if prev_state is None:
                prev_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
            
            q = q.transpose(1, 2).squeeze(2) # (B, H, D)
            k = k.transpose(1, 2).squeeze(2)
            v = v.transpose(1, 2).squeeze(2)
            d = decay.transpose(1, 2).squeeze(2).unsqueeze(-1) # (B, H, 1, 1)
            g = surprise.transpose(1, 2).squeeze(2).unsqueeze(-1) # (B, H, 1, 1)
            
            kv = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
            
            # S_t = decay * S_{t-1} + gate * kv
            current_state = prev_state * d + g * kv
            state_cache.update_state(self.layer_idx, current_state)
            
            y = torch.matmul(q.unsqueeze(2), current_state).squeeze(2)
            y = y.unsqueeze(1)
            
        else:
            # Parallel Mode
            # We use the expansion method O(T D^2) which is efficient for T=1024.
            # S_t = (1-g)*S_{t-1} + g*kv
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            decay = decay.transpose(1, 2) # (B, H, T, 1)
            gate = surprise.transpose(1, 2) # (B, H, T, 1)
            
            kv = torch.einsum('bhtd,bhte->bhtde', k, v)
            
            # Apply gate to KV
            kv = kv * gate.unsqueeze(-1)
            
            # Cumulative sum with decay
            state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
            states = []
            for t in range(T):
                state = state * decay[:, :, t].unsqueeze(-1) + kv[:, :, t]
                states.append(state)
            states = torch.stack(states, dim=2)
            
            y = torch.matmul(q.unsqueeze(-2), states).squeeze(-2)
            y = y.transpose(1, 2).contiguous().view(B, T, -1)

        y = self.c_proj(y)
        y = y + self.mlp(norm(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class RecurrentMemory:
    """
    Recurrent Memory (formerly KV Cache).
    Stores the Fast State for each nesting level.
    """
    def __init__(self, max_batch_size, num_layers, num_heads, head_dim, device, dtype=torch.bfloat16):
        self.states = {} 
        self.max_batch_size = max_batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.is_inference = True
        self.pos = 0

    def get_state(self, layer_idx):
        return self.states.get(layer_idx)

    def update_state(self, layer_idx, new_state):
        self.states[layer_idx] = new_state
        
    def get_pos(self):
        return self.pos
        
    def reset(self):
        self.states = {}
        self.pos = 0

    def prefill(self, other):
        assert self.num_layers == other.num_layers
        for layer_idx, state in other.states.items():
            if self.max_batch_size > other.max_batch_size:
                new_state = state.expand(self.max_batch_size, -1, -1, -1).clone()
            else:
                new_state = state.clone()
            self.states[layer_idx] = new_state
        self.pos = other.pos

class HOPEEnsemble(nn.Module):
    """
    HOPE Ensemble (formerly GPT).
    A Hierarchical Optimizing Processing Ensemble.
    """
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        
        self.continuum = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "levels": nn.ModuleList([NestedAdaptationBlock(config, layer_idx) for layer_idx in range(config.max_layers)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        
        self.meta_controller = MetaController(config)
        
        # Initialize only up to n_layer initially active
        self.active_layers = config.n_layer

    def init_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.zeros_(self.lm_head.weight)
        for block in self.continuum.levels:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.c_proj.weight)
            
        if self.continuum.wte.weight.device.type == "cuda":
            self.continuum.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def get_device(self):
        return self.continuum.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.continuum.wte.weight.numel()
        return 6 * (nparams - nparams_embedding)

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, meta_lr=0.001, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate parameters
        matrix_params = []
        meta_params = []
        embedding_params = list(self.continuum.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        
        for block in self.continuum.levels:
            # Meta-net parameters go to meta_params
            meta_params.extend(list(block.meta_net.parameters()))
            # Other linear params go to matrix_params
            matrix_params.extend(list(block.c_q.parameters()))
            matrix_params.extend(list(block.c_k.parameters()))
            matrix_params.extend(list(block.c_v.parameters()))
            matrix_params.extend(list(block.c_proj.parameters()))
            matrix_params.extend(list(block.mlp.parameters()))
            
        # Meta controller params
        meta_params.extend(list(self.meta_controller.parameters()))

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=meta_params, lr=meta_lr), # Meta parameters get their own LR
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        
        x = self.continuum.wte(idx)
        x = norm(x)
        
        active_layers = self.meta_controller.current_levels
        
        for i in range(active_layers):
            x = x + self.continuum.levels[i](x, state_cache=kv_cache)
            
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
            
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        state_cache = RecurrentMemory(1, self.config.max_layers, self.config.n_head, self.config.n_embd // self.config.n_head, device)
        
        logits = self.forward(ids, kv_cache=state_cache)
        
        for _ in range(max_tokens):
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
            
            logits = self.forward(next_ids, kv_cache=state_cache)
