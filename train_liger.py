import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import tqdm
import os
import tiktoken
import math
import inspect
import datasets
import time
import csv
from typing import List
import argparse

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# Configuration
class Config:
    vocab_size = 50304
    batch_size = 8
    block_size = 512
    eval_interval = 500  # evaluate every N optimizer steps
    learning_rate = 3e-4
    eval_iters = 50  # number of val batches per evaluation
    n_embd = 768
    n_head = 12
    n_layer = 12
    gqa_kv_head = 4
    hidden_size = 2048
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gradient_accumulation_steps = 64  # effective batch = 8 * 64 * 512 = 262k tokens/step
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    max_steps = 50000  # total optimizer steps
    checkpoint_dir = "checkpoints"
    compile = True  # Use torch.compile if available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' == str(device)

config = Config()

def print_banner():
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    tokens_per_step = eff_batch * config.block_size

    print("=" * 60)
    print("  TinyGPT Training Script (Liger Kernel Edition)")
    print("=" * 60)
    print()
    print(f"  Dataset:        Skylion007/openwebtext (streaming)")
    print(f"  Device:         {device_name} ({vram} VRAM)")
    print()
    print("  Model Config:")
    print(f"    Layers:       {config.n_layer}")
    print(f"    Embed dim:    {config.n_embd}")
    print(f"    Heads:        {config.n_head} (KV groups: {config.gqa_kv_head})")
    print(f"    FFN hidden:   {config.hidden_size}")
    print(f"    Context:      {config.block_size} tokens")
    print(f"    Dropout:      {config.dropout}")
    print()
    print("  Training Config:")
    print(f"    Batch size:   {config.batch_size} x {config.gradient_accumulation_steps} accum = {eff_batch} effective")
    print(f"    Tokens/step:  {tokens_per_step:,}")
    print(f"    LR:           {config.learning_rate} -> {8e-4} (OneCycleLR)")
    print(f"    Max steps:    {config.max_steps:,}")
    print(f"    Eval every:   {config.eval_interval} steps")
    print(f"    Compile:      {config.compile}")
    print()
    print("=" * 60)

print_banner()

# Liger Kernel Imports
try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    # from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    # Liger RoPE might be available as an op or we use custom. 
    # Checking for LigerRoPE or similar. If not found, we use custom.
    # For now, we will use custom RoPE as it's architecture specific often.
    # But we definitely use LigerRMSNorm and LigerCrossEntropyLoss.
    print("Liger kernels imported successfully.")
    USE_LIGER = True
except ImportError as e:
    print(f"Error importing Liger kernels: {e}")
    print("Liger kernels not found. Falling back to standard implementations.")
    USE_LIGER = False

# Tokenizer
class Tokenizer:
    def __init__(self, tokenizer_model="gpt2"):
        gpt2_enc = tiktoken.get_encoding(tokenizer_model)
        self.enc = tiktoken.Encoding(
            name=tokenizer_model,
            pat_str=gpt2_enc._pat_str,
            mergeable_ranks=gpt2_enc._mergeable_ranks,
            special_tokens={
                **gpt2_enc._special_tokens,
                "PAD": 50257,
            },
        )
        self.pad_id = self.enc._special_tokens["PAD"]

    def encode(self, s: str) -> List[int]:
        return self.enc.encode(s, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        # Filter out tokens outside tiktoken's vocab (from padded vocab_size)
        valid_tokens = [t for t in tokens if t < self.enc.n_vocab]
        return self.enc.decode(valid_tokens)

tokenizer = Tokenizer()

# Data Loading
def collate_fn(batch):
    texts = [tokenizer.encode(item['text'])[:config.block_size+1] for item in batch]

    input_ids = []
    targets = []

    for text in texts:
        if len(text) <= 1: continue

        inp = text[:-1]
        tgt = text[1:]

        # Padding
        if len(inp) < config.block_size:
            inp = inp + [tokenizer.pad_id] * (config.block_size - len(inp))
        if len(tgt) < config.block_size:
            tgt = tgt + [tokenizer.pad_id] * (config.block_size - len(tgt))

        input_ids.append(torch.tensor(inp, dtype=torch.long))
        targets.append(torch.tensor(tgt, dtype=torch.long))

    if not input_ids: return None

    return {
        'input': torch.stack(input_ids),
        'target': torch.stack(targets)
    }

def prepare_data():
    # Stream OpenWebText — no full download needed
    ds = datasets.load_dataset(
        "Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=False
    )
    ds = ds.shuffle(seed=42, buffer_size=10_000)

    train_loader = DataLoader(
        ds, batch_size=config.batch_size, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )

    return train_loader

# Model Components

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

def get_rms_norm(dim, eps=1e-6):
    if USE_LIGER:
        return LigerRMSNorm(hidden_size=dim, eps=eps)
    return RMSNorm(dim, eps)

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, H, D) or (B, H, T, D) - wait, let's check usage
    # Usually (B, T, H, D) if batch_first=True in logic
    # The notebook implementation assumes (B, T, H, D) for q/k before transpose?
    # Notebook: q = q_rot.transpose(1, 2) -> (B, H, T, head_dim)
    # So input to apply_rotary_emb is (B, T, H, head_dim)
    
    d = x.shape[-1]
    # Reshape to (..., D/2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # freqs_cis: (T, D/2)
    # We need to broadcast. 
    # x_complex: (B, T, H, D/2)
    # freqs_cis: (T, D/2) -> (1, T, 1, D/2)
    
    freqs_cis = freqs_cis[:x.shape[1]] # Crop to seq_len
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
    
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

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
        H = self.n_head
        G = self.n_query_groups
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, G, D)
        v = self.v_proj(x).view(B, T, G, D)

        # Apply RoPE
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # KV Caching
        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=1)
            v = torch.cat([v_past, v], dim=1)

        new_kv_cache = (k, v)

        # Repeat K/V for GQA
        k = k[:, :, :, None, :].expand(B, -1, G, H // G, D).reshape(B, -1, H, D)
        v = v[:, :, :, None, :].expand(B, -1, G, H // G, D).reshape(B, -1, H, D)

        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use causal masking during training (full sequence), disable during cached generation
        # (when using KV cache, q is 1 token attending to all past — no causal mask needed)
        use_causal = is_causal and kv_cache is None
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=use_causal)

        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_output), new_kv_cache

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = get_rms_norm(config.n_embd)
        self.attn = GroupedQueryAttention(config.n_embd, config.n_head, config.gqa_kv_head, config.dropout)
        self.ln2 = get_rms_norm(config.n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(config.n_embd, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, freqs_cis, is_causal=True, kv_cache=None):
        # x: (B, T, C)
        residual = x
        x = self.ln1(x)
        attn_out, new_kv_cache = self.attn(x, freqs_cis, is_causal, kv_cache)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = residual + self.ffwd(x)
        
        return x, new_kv_cache

class GPT(nn.Module):
    def __init__(self, config: Config, pad_id: int = 50257):
        super().__init__()
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # No pos embedding, using RoPE
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = get_rms_norm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.token_embedding.weight = self.lm_head.weight # Weight tying
        
        # Precompute RoPE frequencies
        self.register_buffer('freqs_cis', precompute_freqs_cis(config.n_embd // config.n_head, config.block_size * 2)) 
        # *2 to allow for some extrapolation or just safety
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=None):
        B, T = idx.shape
        device = idx.device
        
        x = self.token_embedding(idx)

        if kv_caches is not None and start_pos is not None:
            # During cached generation: only the new token(s), at the correct RoPE position
            freqs_cis = self.freqs_cis[start_pos:start_pos + T]
        else:
            # During training or initial prompt pass
            freqs_cis = self.freqs_cis[:T]

        is_causal = True

        new_kv_caches = []

        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, freqs_cis, is_causal=is_causal, kv_cache=kv_cache)
            new_kv_caches.append(new_cache)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten for loss calculation
            logits = logits.view(-1, self.config.vocab_size)
            targets = targets.view(-1)
            loss = self.loss_fn(logits, targets)
                
        return logits, loss, new_kv_caches

    @torch.inference_mode()
    @torch.compiler.disable
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, tokenizer=None, stream=False):
        self.eval()
        B, T = idx.shape

        # Initial prefill pass — process the full prompt
        logits, _, kv_caches = self(idx, kv_caches=None, start_pos=None)
        cur_pos = T  # next token position

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if stream and tokenizer:
            print(tokenizer.decode([idx_next.item()]), end="", flush=True)

        # Autoregressive decoding with KV cache — only feed 1 token at a time
        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self(idx_next, kv_caches=kv_caches, start_pos=cur_pos)
            cur_pos += 1
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if stream and tokenizer:
                print(tokenizer.decode([idx_next.item()]), end="", flush=True)

        if stream:
            print()  # newline after streaming

        return idx

# Training Loop
def evaluate(model, train_loader_iter):
    """Run evaluation on a few batches from the stream."""
    model.eval()
    val_loss = 0
    count = 0
    with torch.inference_mode():
        for _ in range(config.eval_iters):
            try:
                batch = next(train_loader_iter)
            except StopIteration:
                break
            if batch is None:
                continue
            v_in = batch['input'].to(config.device, non_blocking=True)
            v_tgt = batch['target'].to(config.device, non_blocking=True)
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                _, v_loss, _ = model(v_in, v_tgt)
            val_loss += v_loss.item()
            count += 1
    model.train()
    return val_loss / max(count, 1)


def find_latest_checkpoint():
    """Find the most recent checkpoint in the checkpoint directory."""
    if not os.path.exists(config.checkpoint_dir):
        return None
    ckpts = [f for f in os.listdir(config.checkpoint_dir) if f.startswith("ckpt_step") and f.endswith(".pth")]
    if not ckpts:
        return None
    # Extract step numbers and find the latest
    ckpts.sort(key=lambda f: int(f.replace("ckpt_step", "").replace(".pth", "")))
    return os.path.join(config.checkpoint_dir, ckpts[-1])


def train(resume=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize CSV file for logging (append if resuming)
    csv_path = os.path.join(config.checkpoint_dir, "training_metrics.csv")
    existing_ckpt = find_latest_checkpoint()
    if resume and not existing_ckpt:
        print("--resume passed but no checkpoint found. Starting from scratch.")
    if not resume and existing_ckpt:
        answer = input(f"Found existing checkpoint: {existing_ckpt}\nResume training? [y/N]: ").strip().lower()
        resume = answer in ("y", "yes")
    resume_ckpt = existing_ckpt if resume else None
    if resume_ckpt:
        csv_file = open(csv_path, 'a', newline='')
    else:
        csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    if not resume_ckpt:
        csv_writer.writerow(['opt_step', 'train_loss', 'val_loss', 'train_perplexity', 'val_perplexity',
                             'learning_rate', 'tokens_per_sec', 'avg_tokens_per_sec', 'total_tokens_processed', 'elapsed_time', 'generated_text'])

    train_loader = prepare_data()
    train_iter = iter(train_loader)

    # Separate validation stream (re-open without shuffle so it's deterministic)
    val_ds = datasets.load_dataset(
        "Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=False
    ).skip(8_000_000)  # skip ahead to use different data for val
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, collate_fn=collate_fn,
        num_workers=1, pin_memory=True
    )
    val_iter = iter(val_loader)

    model = GPT(config, pad_id=tokenizer.pad_id).to(config.device)

    if USE_LIGER:
        print("Using Liger kernels for RMSNorm.")
    else:
        print("Using standard PyTorch implementations.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(config.beta1, config.beta2),
        fused=config.use_fused
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=8e-4, total_steps=config.max_steps, pct_start=0.05
    )

    # --- Resume from checkpoint if available ---
    start_opt_step = 0
    total_tokens_processed = 0
    train_losses = []
    val_losses = []

    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=config.device, weights_only=False)
        # Strip _orig_mod. prefix added by torch.compile
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_opt_step = ckpt['opt_step']
        total_tokens_processed = ckpt.get('total_tokens_processed', 0)
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        print(f"Resumed at optimizer step {start_opt_step}, {total_tokens_processed:,} tokens processed so far")
    else:
        print("No checkpoint found, starting from scratch.")

    if config.compile:
        print("Compiling model...")
        model = torch.compile(model)
        print("Model compiled.")

    total_micro_steps = config.max_steps * config.gradient_accumulation_steps
    print(f"Starting training on {config.device} with Liger={USE_LIGER}")
    print(f"Optimizer steps: {config.max_steps}, Micro-steps: {total_micro_steps}")

    model.train()

    # Progress bar tracks optimizer steps
    pbar = tqdm.tqdm(total=config.max_steps, initial=start_opt_step, desc="Training")

    # Throughput tracking
    start_time = time.time()
    micro_step = 0
    opt_step = start_opt_step
    running_loss = 0.0

    while opt_step < config.max_steps:
        # --- Micro-step ---
        try:
            batch = next(train_iter)
        except StopIteration:
            # Stream exhausted — restart
            train_loader = prepare_data()
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if batch is None:
            continue

        t0 = time.time()

        inputs = batch['input'].to(config.device, non_blocking=True)
        targets = batch['target'].to(config.device, non_blocking=True)

        num_tokens = inputs.shape[0] * inputs.shape[1]
        total_tokens_processed += num_tokens

        with torch.autocast(device_type=config.device, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            _, loss, _ = model(inputs, targets)

        original_loss = loss.item()
        running_loss += original_loss
        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        train_losses.append(original_loss)
        if len(train_losses) > 10000:
            train_losses = train_losses[-5000:]

        micro_step += 1

        # --- Optimizer step ---
        if micro_step % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            opt_step += 1
            pbar.update(1)

            avg_micro_loss = running_loss / config.gradient_accumulation_steps
            running_loss = 0.0

            t1 = time.time()
            tokens_per_sec = (num_tokens * config.gradient_accumulation_steps) / max(t1 - t0, 1e-6)
            pbar.set_description(f"Loss: {avg_micro_loss:.4f} | {tokens_per_sec:.0f} tok/s")

            # --- Evaluation & checkpoint ---
            if opt_step % config.eval_interval == 0:
                val_loss = evaluate(model, val_iter)
                val_losses.append(val_loss)

                train_perplexity = math.exp(min(avg_micro_loss, 20))
                val_perplexity = math.exp(min(val_loss, 20))

                elapsed_time = time.time() - start_time
                avg_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
                current_lr = optimizer.param_groups[0]['lr']

                print(f"\n[Step {opt_step}] Train Loss {avg_micro_loss:.4f}, Val Loss {val_loss:.4f}")
                print(f"Train PPL: {train_perplexity:.2f}, Val PPL: {val_perplexity:.2f}")
                print(f"Elapsed: {elapsed_time:.1f}s, Avg throughput: {avg_tokens_per_sec:.0f} tok/s")
                print(f"Total tokens: {total_tokens_processed:,}")

                # Generate sample text
                with torch.inference_mode():
                    prompt = "The meaning of life is"
                    prompt_tokens = tokenizer.encode(prompt)
                    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(config.device)
                    print(f"\nGenerated: {prompt}", end="", flush=True)
                    generated = model.generate(prompt_tensor, max_new_tokens=50, temperature=0.8, top_k=40, tokenizer=tokenizer, stream=True)
                    generated_text = tokenizer.decode(generated[0].tolist())
                    print()  # blank line after generation

                csv_writer.writerow([opt_step, avg_micro_loss, val_loss, train_perplexity, val_perplexity,
                                    current_lr, tokens_per_sec, avg_tokens_per_sec, total_tokens_processed, elapsed_time, generated_text])
                csv_file.flush()

                # Save checkpoint
                ckpt_path = os.path.join(config.checkpoint_dir, f"ckpt_step{opt_step}.pth")
                checkpoint = {
                    'opt_step': opt_step,
                    'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_micro_loss,
                    'val_loss': val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'total_tokens_processed': total_tokens_processed,
                    'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')}
                }
                torch.save(checkpoint, ckpt_path)
                print(f"Checkpoint saved to {ckpt_path}")

                model.train()

    pbar.close()
    csv_file.close()
    print(f"\nTraining complete. Metrics saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
