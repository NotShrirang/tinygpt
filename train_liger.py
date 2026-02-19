import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import tqdm
import os
import math
import inspect
import datasets
import time
import csv
import argparse

from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer
from tinygpt.layers import USE_LIGER_RMS as USE_LIGER

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# Configuration — model params come from TinyGPT2Config, training params here
model_config = TinyGPT2Config()

class Config:
    # Model config (delegate to TinyGPT2Config)
    vocab_size = model_config.vocab_size
    block_size = model_config.block_size
    n_embd = model_config.n_embd
    n_head = model_config.n_head
    n_layer = model_config.n_layer
    gqa_kv_head = model_config.gqa_kv_head
    hidden_size = model_config.hidden_size
    dropout = model_config.dropout
    weight_decay = model_config.weight_decay
    beta1 = model_config.beta1
    beta2 = model_config.beta2

    # Training config
    batch_size = 8
    eval_interval = 500  # evaluate every N optimizer steps
    learning_rate = 3e-4
    eval_iters = 50  # number of val batches per evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gradient_accumulation_steps = 64  # effective batch = 8 * 64 * 512 = 262k tokens/step
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

if USE_LIGER:
    print("Liger kernels imported successfully.")
else:
    print("Liger kernels not found. Falling back to standard implementations.")

# Tokenizer
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

    model = TinyGPT2(model_config, pad_id=tokenizer.pad_id).to(config.device)

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
