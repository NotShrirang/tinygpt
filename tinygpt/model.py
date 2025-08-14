import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Check if it is linux
if os.name == 'posix':
    from liger_kernel.transformers import LigerSwiGLUMLP, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss

from tinygpt.config import GPTConfig, MoEGPTConfig
from tinygpt.layers import DecoderBlock, CausalMoEBlock, RotaryEmbeddings
from tinygpt.utils import remove_orig_mod_prefix, map_swiglu_keys


class GPTLanguageModel(nn.Module):
    """
    A simple GPT language model with a stack of transformer blocks.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(self.n_layer)])
        
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.apply(self._init_weights)

        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    @classmethod
    def from_pretrained(self, pretrained_model_path: str, device: str = "cpu") -> "GPTLanguageModel":
        """
        Load a pretrained model from the specified path.
        """
        model = self(GPTConfig())
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Given a sequence of indices 'idx', generate 'max_new_tokens' new tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class MoEGPTLanguageModel(nn.Module):
    def __init__(self, config: MoEGPTConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.ModuleList([CausalMoEBlock(self.config, self.device) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.block_size = self.config.block_size

        if self.config.pad_token_id:
            self.loss_fct = LigerFusedLinearCrossEntropyLoss(ignore_index=self.config.pad_token_id).to(self.lm_head.weight.device)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, inference=False):
        B,T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(
            torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if inference or targets is None:
            return logits if inference else (logits, None)
        logits_flat = logits.view(-1, self.config.vocab_size)
        target_flat = targets.view(-1)
        if hasattr(self, 'loss_fct'):
            loss = self.loss_fct(self.lm_head.weight,
                                 x.view(-1, self.config.n_embd),
                                 target_flat)
        else:
            loss = F.cross_entropy(logits_flat, target_flat, ignore_index=self.config.pad_token_id)
        return logits, loss
    
    @classmethod
    def from_pretrained(self, pretrained_model_path: str, device: str = "cpu") -> "MoEGPTLanguageModel":
        """
        Load a pretrained model from the specified path.
        """
        model = self(MoEGPTConfig(), device=device)
        state_dict = torch.load(pretrained_model_path, map_location=device)
        state_dict = remove_orig_mod_prefix(state_dict)
        state_dict = map_swiglu_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")

        model.eval()
        model.to(device)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits,_ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
