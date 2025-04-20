import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Subset
import tqdm
import json
import datasets
from typing import List
import os
import pandas as pd
import tiktoken
import inspect

encoding = tiktoken.get_encoding("gpt2")

class Tokenizer:
    def __init__(self, tokenizer_model="gpt2"):
        self.enc = tiktoken.get_encoding(tokenizer_model)
        self.tokenizer_model = tokenizer_model

        self.n_words = self.enc.n_vocab
        self.bos_id = None
        self.eos_id = self.enc.eot_token
        self.pad_id = None

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        t = self.enc.encode(s)
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
    

tokenizer = Tokenizer(tokenizer_model="gpt2")

vocab_size = 50304
batch_size = 32
block_size = 512
max_iters = 1
eval_interval = 1000
learning_rate = 7e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 256
n_embd = 512
n_head = 8
n_layer = 8
dropout = 0.3

target_batch_size = 8192 * 2
gradient_accumulation_steps = target_batch_size // batch_size
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode(s): return tokenizer.encode(s, bos=False, eos=False)

def decode(l):
	try:
		return tokenizer.decode(l)
	except:
		return ""
     
ds = datasets.load_dataset("roneneldan/TinyStories")

ds = ds.with_format("torch")

def collate_fn(batch):
    texts = [encode(item['text'])[:block_size] for item in batch]  # Truncate to block_size
    padded_texts = [t + [0] * (block_size - len(t)) for t in texts]  # Pad to 512
    return {
        'text': torch.tensor(padded_texts, dtype=torch.long)
    }


subset_indices = list(range(eval_iters))
dataset_valid = Subset(ds['validation'], subset_indices)

train_dataloader = DataLoader(ds['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, collate_fn=collate_fn)

def generate_square_subsequent_mask(sz):
    """
    Generates a causal (upper-triangular) mask for a sequence of length 'sz'.
    Positions with True (or -inf when using additive masks) will be masked.
    Here, we create an additive mask with -inf for masked positions.
    """
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

class Block(nn.Module):
    """Transformer block using PyTorch's MultiheadAttention with an explicit causal mask."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        # PyTorch's MultiheadAttention
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True  # Expect input as (batch, seq, feature)
        )
        
        # Feed-forward network
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
        # Layer normalization layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # x has shape (B, T, C)
        T = x.size(1)
        
        # Pre-LayerNorm for attention
        x_ln = self.ln1(x)
        # Create a causal mask explicitly for the current sequence length
        causal_mask = generate_square_subsequent_mask(T).to(x.device)
        
        # Self-attention: note that we pass attn_mask instead of is_causal
        attn_output, _ = self.attn(
            query=x_ln,
            key=x_ln,
            value=x_ln,
            attn_mask=causal_mask,  # Using the explicit causal mask here
            need_weights=False
        )
        x = x + attn_output
        
        # Feed-forward block with pre-LayerNorm
        x = x + self.ffwd(self.ln2(x))
        
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        
        # Final layer normalization and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
		# Initialize weights for Linear and Embedding layers
        self.apply(self._init_weights)

        # Weight tying: share the weight matrix between token embeddings and the output projection
        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Obtain token embeddings and add positional embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, C)
            
        # Final layer normalization and output projection to logits
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Given a sequence of indices 'idx', generate 'max_new_tokens' new tokens.
        """
        for _ in range(max_new_tokens):
            # Crop the sequence to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
    
torch.cuda.empty_cache()

model = GPTLanguageModel()

# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)

model = model.to(device)
model = torch.compile(model)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and 'cuda' == str(device)
print(f"{use_fused=}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), eps=1e-8, fused=use_fused)

# T_max = len(train_dataloader)
# warmup_steps = 0.01 * T_max
# scheduler = lr_scheduler.OneCycleLR(
#     optimizer, max_lr=4e-4, total_steps=T_max, pct_start=0.01
# )
true_total_steps = len(train_dataloader) // gradient_accumulation_steps
scheduler = lr_scheduler.OneCycleLR(
    optimizer, max_lr=8e-5, total_steps=true_total_steps, pct_start=0.05
)

os.makedirs("ckpt/", exist_ok=True)

sample = tokenizer.decode(tokenizer.encode(ds["train"][0]["text"][:100], bos=True, eos=True))
print(f"Sample text: {sample}")

def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, loss = model(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx

# print(gradient_accumulation_steps, batch_size, target_batch_size)
print(f"{gradient_accumulation_steps=}, {batch_size=}, {target_batch_size=}")

with open("losses.txt", "w") as f:
	f.write("Training Loss,Validation Loss,Output\n")
     

for iter, batch in enumerate(tqdm.notebook.tqdm(train_dataloader, total=len(train_dataloader))):
    inputs, targets = batch['text'], batch['text']
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
        logits, loss = model(inputs, targets)

    loss = loss / gradient_accumulation_steps
    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    if (iter + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if iter % (gradient_accumulation_steps * 2) == 0 or iter == max_iters - 1:
        print(f"\nStep {iter}: Performing validation")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        model.eval()
        with torch.no_grad():
            val_loss = 0
            train_loss = loss.item() * gradient_accumulation_steps
            for batch in tqdm.notebook.tqdm(valid_dataloader, total=len(valid_dataloader)):
                inputs, targets = batch['text'], batch['text']
                inputs, targets = inputs.to(device), targets.to(device)
                _, loss = model(inputs, targets)
                val_loss += loss.item()

            torch.save(model.state_dict(), f"ckpt/ckpt_{iter}.pt")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Validation loss: {val_loss / len(valid_dataloader):.4f}")

            prompt = "One day, a "
            prompt = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            output = decode(generate(model, prompt, max_new_tokens=50)[0].tolist())
            print(output)
            with open("losses.txt", "a") as f:
                f.write(f"{train_loss},{val_loss / len(valid_dataloader)},\"{output}\"\n")
        model.train()

torch.save(model.state_dict(), "ckpt/ckpt_final.pt")
model = model.eval()

prompt = "There was a girl who"
prompt = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
print(decode(generate(model, prompt, max_new_tokens=50)[0].tolist()))

prompt = "One day, a little girl named Lily found"
prompt = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
print(decode(generate(model, prompt, max_new_tokens=50)[0].tolist()))