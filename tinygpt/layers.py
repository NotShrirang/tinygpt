from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Check if CUDA is available and we're on a POSIX system before importing liger_kernel
USE_LIGER = os.name == 'posix' and torch.cuda.is_available()

if USE_LIGER:
    try:
        from liger_kernel.transformers import LigerSwiGLUMLP, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
    except ImportError:
        USE_LIGER = False

from tinygpt.utils import generate_square_subsequent_mask, generate_square_subsequent_mask_with_device
from tinygpt.config import GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig


class DecoderBlock(nn.Module):
    """Transformer block using PyTorch's MultiheadAttention with an explicit causal mask."""
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_embd = config.n_embd
        n_head = config.n_head
        dropout = config.dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        T = x.size(1)

        x_ln = self.ln1(x)

        causal_mask = generate_square_subsequent_mask(T).to(x.device)

        attn_output, _ = self.attn(
            query=x_ln,
            key=x_ln,
            value=x_ln,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_output

        x = x + self.ffwd(self.ln2(x))
        
        return x


class RotaryEmbeddings(nn.Module):
    def __init__(self, config: MoEGPTConfig, device):
        super().__init__()
        self.dim = config.n_embd // config.n_head
        self.device = device

    def forward(self, x, q=None, k=None):
        # assume q, k passed; else x used
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=self.device).unsqueeze(1)
        freqs = 10000 ** (-torch.arange(0, self.dim, 2, device=self.device) / self.dim)
        angles = pos * freqs.unsqueeze(0)
        cos, sin = angles.cos(), angles.sin()
        def apply(t):
            t = t.view(t.size(0), t.size(1), -1, 2)
            t2 = torch.stack([t[...,0]*cos - t[...,1]*sin,
                              t[...,1]*cos + t[...,0]*sin], dim=-1)
            return t2.view_as(t)
        return apply(q), apply(k)


class SWiGLUExpertMoE(nn.Module):
    def __init__(self, config: MoEGPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        hidden = self.config.n_embd * 2
        
        if USE_LIGER:
            @dataclass
            class Cfg:
                hidden_size = self.config.n_embd
                intermediate_size = hidden
                hidden_act = 'swish'
            self.swiglu = LigerSwiGLUMLP(Cfg())
        else:
            # Fallback SwiGLU implementation
            self.gate_proj = nn.Linear(self.config.n_embd, hidden, bias=False)
            self.up_proj = nn.Linear(self.config.n_embd, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, self.config.n_embd, bias=False)
    
    def forward(self, x):
        if USE_LIGER:
            return self.swiglu(x)
        else:
            # Manual SwiGLU implementation
            gate = F.silu(self.gate_proj(x))  # SiLU activation
            up = self.up_proj(x)
            return self.down_proj(gate * up)


class MoeLayer(nn.Module):
    def __init__(self, config: MoEGPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device

        self.experts = nn.ModuleList([SWiGLUExpertMoE(config, device) for _ in range(self.config.n_experts)])
        self.gate = nn.Linear(self.config.n_embd, self.config.n_experts, bias=False, device=self.device)
        if self.config.noisy_topk and not self.config.use_checkpointing:
            self.noise = nn.Linear(self.config.n_embd, self.config.n_experts, bias=False, device=self.device)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        gate_logits = self.gate(x)  # [B, T, E]
        if self.config.noisy_topk and not self.config.use_checkpointing:
            noise = F.softplus(self.noise(x)) * torch.randn_like(gate_logits)
            gate_logits = gate_logits + noise
        # pick top-K experts indices and probs
        topk_vals, topk_idx = torch.topk(gate_logits, self.config.top_experts, dim=-1)  # both [B, T, K]
        probs = F.softmax(topk_vals, dim=-1)  # [B, T, K]
        out = torch.zeros_like(x)
        # for each expert, gather contributions
        for e in range(self.config.n_experts):
            # mask tokens assigned to expert e across any top-k slot
            mask2d = (topk_idx == e).any(dim=-1)  # [B, T]
            if not mask2d.any():
                continue
            # weights per token for this expert
            w = (probs * (topk_idx == e).float()).sum(dim=-1)  # [B, T]
            # select input tokens
            x_sel = x[mask2d]
            # run through expert
            y_sel = self.experts[e](x_sel)
            # weight outputs and scatter back
            out[mask2d] += y_sel * w[mask2d].unsqueeze(-1)
        return out


class CausalMoEBlock(nn.Module):
    def __init__(self, config: MoEGPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.rotary = RotaryEmbeddings(self.config, self.device)
        self.ln1 = nn.LayerNorm(self.config.n_embd)
        self.ln2 = nn.LayerNorm(self.config.n_embd)
        self.moe = MoeLayer(self.config, self.device)

    def forward(self, x):
        B,T,C = x.shape
        x1 = self.ln1(x)
        # split heads
        qkv = x1
        attn_mask = generate_square_subsequent_mask_with_device(T, device=x.device)
        attn_out,_ = self.attn(qkv, qkv, qkv,
                               attn_mask=attn_mask,
                               need_weights=False)
        x = x + attn_out
        x = x + self.moe(self.ln2(x))
        return x


# Wikipedia MoE specific components
class WikipediaRotaryEmbeddings(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x, q=None, k=None):
        # assume q, k passed; else x used
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=self.device).unsqueeze(1)
        freqs = 10000 ** (-torch.arange(0, self.dim, 2, device=self.device) / self.dim)
        angles = pos * freqs.unsqueeze(0)
        cos, sin = angles.cos(), angles.sin()
        def apply(t):
            t = t.view(t.size(0), t.size(1), -1, 2)
            t2 = torch.stack([t[...,0]*cos - t[...,1]*sin,
                              t[...,1]*cos + t[...,0]*sin], dim=-1)
            return t2.view_as(t)
        return apply(q), apply(k)


class WikipediaSWiGLUExpertMoE(nn.Module):
    def __init__(self, config: WikipediaMoEGPTConfig):
        super().__init__()
        hidden = config.n_embd * 2
        
        if USE_LIGER:
            @dataclass
            class Cfg:
                hidden_size = config.n_embd
                intermediate_size = hidden
                hidden_act = 'swish'
            self.swiglu = LigerSwiGLUMLP(Cfg())
        else:
            # Fallback SwiGLU implementation
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=False)
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=False)
    
    def forward(self, x):
        if USE_LIGER:
            return self.swiglu(x)
        else:
            # Manual SwiGLU implementation
            gate = F.silu(self.gate_proj(x))  # SiLU activation
            up = self.up_proj(x)
            return self.down_proj(gate * up)


class WikipediaMoeLayer(nn.Module):
    def __init__(self, config: WikipediaMoEGPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.experts = nn.ModuleList([WikipediaSWiGLUExpertMoE(config) for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False, device=device)
        if config.noisy_topk and not config.use_checkpointing:
            self.noise = nn.Linear(config.n_embd, config.n_experts, bias=False, device=device)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        gate_logits = self.gate(x)  # [B, T, E]
        if self.config.noisy_topk and not self.config.use_checkpointing:
            noise = F.softplus(self.noise(x)) * torch.randn_like(gate_logits)
            gate_logits = gate_logits + noise
        # pick top-K experts indices and probs
        topk_vals, topk_idx = torch.topk(gate_logits, self.config.top_experts, dim=-1)  # both [B, T, K]
        probs = F.softmax(topk_vals, dim=-1)  # [B, T, K]
        out = torch.zeros_like(x)
        # for each expert, gather contributions
        for e in range(self.config.n_experts):
            # mask tokens assigned to expert e across any top-k slot
            mask2d = (topk_idx == e).any(dim=-1)  # [B, T]
            if not mask2d.any():
                continue
            # weights per token for this expert
            w = (probs * (topk_idx == e).float()).sum(dim=-1)  # [B, T]
            # select input tokens
            x_sel = x[mask2d]
            # run through expert
            y_sel = self.experts[e](x_sel)
            # weight outputs and scatter back
            out[mask2d] += y_sel * w[mask2d].unsqueeze(-1)
        return out


class WikipediaCausalMoEBlock(nn.Module):
    def __init__(self, config: WikipediaMoEGPTConfig, device):
        super().__init__()
        self.config = config
        self.device = device
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head,
                                          dropout=config.dropout,
                                          batch_first=True)
        self.rotary = WikipediaRotaryEmbeddings(config.n_embd//config.n_head, device)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.moe = WikipediaMoeLayer(config, device)

    def forward(self, x):
        B,T,C = x.shape
        x1 = self.ln1(x)
        # split heads
        qkv = x1
        attn_mask = generate_square_subsequent_mask_with_device(T, device=x.device)
        attn_out,_ = self.attn(qkv, qkv, qkv,
                               attn_mask=attn_mask,
                               need_weights=False)
        x = x + attn_out
        x = x + self.moe(self.ln2(x))
        return x
