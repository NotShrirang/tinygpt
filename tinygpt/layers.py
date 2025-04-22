import torch
import torch.nn as nn

from tinygpt.utils import generate_square_subsequent_mask
from tinygpt.config import GPTConfig


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
