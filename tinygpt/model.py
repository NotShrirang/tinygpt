import torch
import torch.nn as nn
import torch.nn.functional as F

from tinygpt.config import GPTConfig
from tinygpt.layers import DecoderBlock


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
