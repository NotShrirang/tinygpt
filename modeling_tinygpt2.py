"""HuggingFace-compatible model definition for TinyGPT2.

This file is self-contained so it works when downloaded from the HuggingFace Hub
with `trust_remote_code=True`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_tinygpt2 import TinyGPT2HFConfig


# ---------------------------------------------------------------------------
# Layers (self-contained copies so this file works standalone on HF Hub)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, H, D)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1]].view(1, x.shape[1], 1, -1)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_query_groups, dropout=0.1):
        super().__init__()
        assert n_head % n_query_groups == 0
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_query_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, n_query_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, is_causal=True, kv_cache=None):
        B, T, C = x.shape
        H, G, D = self.n_head, self.n_query_groups, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, G, D)
        v = self.v_proj(x).view(B, T, G, D)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=1)
            v = torch.cat([v_past, v], dim=1)

        new_kv_cache = (k, v)

        k = k[:, :, :, None, :].expand(B, -1, G, H // G, D).reshape(B, -1, H, D)
        v = v[:, :, :, None, :].expand(B, -1, G, H // G, D).reshape(B, -1, H, D)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        use_causal = is_causal and kv_cache is None
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output), new_kv_cache


class TinyGPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(
            config.n_embd, config.n_head, config.gqa_kv_head, config.dropout
        )
        self.ln2 = RMSNorm(config.n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, freqs_cis, is_causal=True, kv_cache=None):
        residual = x
        x = self.ln1(x)
        attn_out, new_kv_cache = self.attn(x, freqs_cis, is_causal, kv_cache)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        x = residual + self.ffwd(x)
        return x, new_kv_cache


# ---------------------------------------------------------------------------
# HuggingFace PreTrainedModel wrapper
# ---------------------------------------------------------------------------

class TinyGPT2ForCausalLM(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}
    config_class = TinyGPT2HFConfig

    def __init__(self, config: TinyGPT2HFConfig):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [TinyGPT2Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.n_embd // config.n_head, config.block_size * 2
            ),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        use_cache=False,
        **kwargs,
    ):
        B, T = input_ids.shape

        x = self.token_embedding(input_ids)

        if past_key_values is not None and len(past_key_values) > 0:
            start_pos = past_key_values[0][0].shape[1]  # length of cached keys
            freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        else:
            freqs_cis = self.freqs_cis[:T]

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = past_key_values[i] if past_key_values else None
            x, new_cache = block(x, freqs_cis, is_causal=True, kv_cache=kv_cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_kv_caches if use_cache else None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            (k.index_select(0, beam_idx), v.index_select(0, beam_idx))
            for k, v in past_key_values
        )
