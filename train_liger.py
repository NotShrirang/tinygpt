import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import tqdm
import math
import inspect
import datasets
import time
import csv
import argparse
import random
import bitsandbytes as bnb

from tinygpt import TinyGPT2, TinyGPT2Config, TinyGPT2_1Config, Tokenizer
from tinygpt.layers import USE_LIGER_RMS as USE_LIGER

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

# Random validation prompts pool
EVAL_PROMPTS = [
    "The meaning of life is",
    "In the year 2050, technology will",
    "The most important scientific discovery was",
    "Once upon a time in a distant land",
    "The history of mathematics begins with",
    "Climate change is caused by",
    "The best way to learn programming is",
    "When the sun sets over the ocean",
    "Artificial intelligence will transform",
    "The theory of relativity explains",
    "In a small village near the mountains",
    "The human brain is capable of",
    "Democracy is a system of government where",
    "The universe began approximately",
    "To write a good essay, you should",
    "The capital of France is",
    "Photosynthesis is the process by which",
    "Shakespeare wrote many plays including",
    "The first computer was invented",
    "Music has the power to",
]


def get_config(config_name):
    """Get model config and training hyperparams by name."""
    if config_name == "v2.1":
        return TinyGPT2_1Config(), {
            'batch_size': 12,
            'gradient_accumulation_steps': 128,
            'max_steps': 30500,
            'max_lr': 6e-4,
            'learning_rate': 3e-4,
            'eval_interval': 500,
            'eval_iters': 50,
            'checkpoint_dir': 'checkpoints_v2.1',
            'gradient_checkpointing': True,
        }
    else:  # default — TinyGPT2 (95M)
        return TinyGPT2Config(), {
            'batch_size': 8,
            'gradient_accumulation_steps': 64,
            'max_steps': 50000,
            'max_lr': 8e-4,
            'learning_rate': 3e-4,
            'eval_interval': 500,
            'eval_iters': 50,
            'checkpoint_dir': 'checkpoints',
            'gradient_checkpointing': False,
        }


class Config:
    """Training configuration built from model config + training hyperparams."""
    def __init__(self, model_config, train_params):
        # Model config
        self.vocab_size = model_config.vocab_size
        self.block_size = model_config.block_size
        self.n_embd = model_config.n_embd
        self.n_head = model_config.n_head
        self.n_layer = model_config.n_layer
        self.gqa_kv_head = model_config.gqa_kv_head
        self.hidden_size = model_config.hidden_size
        self.dropout = model_config.dropout
        self.weight_decay = model_config.weight_decay
        self.beta1 = model_config.beta1
        self.beta2 = model_config.beta2

        # Training config
        self.batch_size = train_params['batch_size']
        self.eval_interval = train_params['eval_interval']
        self.learning_rate = train_params['learning_rate']
        self.max_lr = train_params['max_lr']
        self.eval_iters = train_params['eval_iters']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gradient_accumulation_steps = train_params['gradient_accumulation_steps']
        self.max_steps = train_params['max_steps']
        self.checkpoint_dir = train_params['checkpoint_dir']
        self.gradient_checkpointing = train_params['gradient_checkpointing']
        self.compile = True


def print_banner(config, config_name):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    tokens_per_step = eff_batch * config.block_size

    print("=" * 60)
    print(f"  TinyGPT Training Script (Liger Kernel Edition)")
    print(f"  Model: TinyGPT2 {'v2.1 (~183M)' if config_name == 'v2.1' else 'default (~95M)'}")
    print("=" * 60)
    print()
    print(f"  Dataset:        HuggingFaceFW/fineweb-edu (streaming)")
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
    print(f"    LR:           {config.learning_rate} -> {config.max_lr} (OneCycleLR)")
    print(f"    Optimizer:    8-bit AdamW (bitsandbytes)")
    print(f"    Max steps:    {config.max_steps:,}")
    print(f"    Eval every:   {config.eval_interval} steps")
    print(f"    Compile:      {config.compile}")
    print(f"    Grad ckpt:    {config.gradient_checkpointing}")
    print()
    print("=" * 60)


# Tokenizer
tokenizer = Tokenizer()

# Data Loading
def collate_fn(batch, block_size):
    texts = [tokenizer.encode(item['text'])[:block_size+1] for item in batch]

    input_ids = []
    targets = []

    for text in texts:
        if len(text) <= 1: continue

        inp = text[:-1]
        tgt = text[1:]

        # Padding
        if len(inp) < block_size:
            inp = inp + [tokenizer.pad_id] * (block_size - len(inp))
        if len(tgt) < block_size:
            tgt = tgt + [tokenizer.pad_id] * (block_size - len(tgt))

        input_ids.append(torch.tensor(inp, dtype=torch.long))
        targets.append(torch.tensor(tgt, dtype=torch.long))

    if not input_ids: return None

    return {
        'input': torch.stack(input_ids),
        'target': torch.stack(targets)
    }

def prepare_data(config):
    # Stream FineWeb-Edu — no full download needed
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True, trust_remote_code=False
    )
    ds = ds.shuffle(seed=42, buffer_size=1_000)

    train_loader = DataLoader(
        ds, batch_size=config.batch_size,
        collate_fn=lambda batch: collate_fn(batch, config.block_size),
        num_workers=0, pin_memory=True
    )

    return train_loader

# Training Loop
def evaluate(model, val_iter, config):
    """Run evaluation on a few batches from the stream."""
    model.eval()
    val_loss = 0
    count = 0
    with torch.inference_mode():
        for _ in range(config.eval_iters):
            try:
                batch = next(val_iter)
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


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    # Check for emergency checkpoint first, then step checkpoints
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not ckpts:
        return None
    # Prefer step checkpoints over emergency
    step_ckpts = [f for f in ckpts if f.startswith("ckpt_step")]
    if step_ckpts:
        step_ckpts.sort(key=lambda f: int(f.replace("ckpt_step", "").replace(".pth", "")))
        return os.path.join(checkpoint_dir, step_ckpts[-1])
    # Fall back to emergency checkpoint
    if "ckpt_emergency.pth" in ckpts:
        return os.path.join(checkpoint_dir, "ckpt_emergency.pth")
    return None


def train(resume=False, config_name="default"):
    model_config, train_params = get_config(config_name)
    config = Config(model_config, train_params)

    print_banner(config, config_name)

    if USE_LIGER:
        print("Liger kernels imported successfully.")
    else:
        print("Liger kernels not found. Falling back to standard implementations.")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize CSV file for logging (append if resuming)
    csv_path = os.path.join(config.checkpoint_dir, "training_metrics.csv")
    existing_ckpt = find_latest_checkpoint(config.checkpoint_dir)
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

    train_loader = prepare_data(config)
    train_iter = iter(train_loader)

    # Separate validation stream (skip ahead to use different data for val)
    val_ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", name="default", split="train", streaming=True, trust_remote_code=False
    ).skip(8_000_000)
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        collate_fn=lambda batch: collate_fn(batch, config.block_size),
        num_workers=0, pin_memory=True
    )
    val_iter = iter(val_loader)

    model = TinyGPT2(model_config, pad_id=tokenizer.pad_id).to(config.device)

    # Enable gradient checkpointing for large models
    if config.gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled.")

    if USE_LIGER:
        print("Using Liger kernels for RMSNorm.")
    else:
        print("Using standard PyTorch implementations.")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    # 8-bit AdamW optimizer (bitsandbytes) — halves optimizer memory
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(config.beta1, config.beta2),
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.max_lr, total_steps=config.max_steps, pct_start=0.05
    )

    # --- Resume from checkpoint if available ---
    start_opt_step = 0
    total_tokens_processed = 0
    train_losses = []
    val_losses = []

    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        # Load to CPU first to avoid VRAM spike
        ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt['model_state_dict']
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_opt_step = ckpt['opt_step']
        total_tokens_processed = ckpt.get('total_tokens_processed', 0)
        train_losses = ckpt.get('train_losses', [])
        val_losses = ckpt.get('val_losses', [])
        del ckpt
        torch.cuda.empty_cache()
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

    def save_checkpoint(step, loss_val=None, val_loss_val=None, emergency=False):
        """Save a checkpoint. Used for regular saves and emergency exits."""
        tag = "emergency" if emergency else f"step{step}"
        ckpt_path = os.path.join(config.checkpoint_dir, f"ckpt_{tag}.pth")
        ckpt = {
            'opt_step': step,
            'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': loss_val,
            'val_loss': val_loss_val,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'total_tokens_processed': total_tokens_processed,
            'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')}
        }
        torch.save(ckpt, ckpt_path)
        print(f"\n{'Emergency checkpoint' if emergency else 'Checkpoint'} saved to {ckpt_path}")

    try:
        while opt_step < config.max_steps:
            # --- Micro-step ---
            try:
                batch = next(train_iter)
            except StopIteration:
                # Stream exhausted — restart
                train_loader = prepare_data(config)
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
                    val_loss = evaluate(model, val_iter, config)
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

                    # Generate sample text with random prompt
                    with torch.inference_mode():
                        prompt = random.choice(EVAL_PROMPTS)
                        prompt_tokens = tokenizer.encode(prompt)
                        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(config.device)
                        print(f"\nGenerated: {prompt}", end="", flush=True)
                        generated = model.generate(prompt_tensor, max_new_tokens=50, temperature=0.8, top_k=40, tokenizer=tokenizer, stream=True)
                        generated_text = tokenizer.decode(generated[0].tolist())
                        print()  # blank line after generation

                    csv_writer.writerow([opt_step, avg_micro_loss, val_loss, train_perplexity, val_perplexity,
                                        current_lr, tokens_per_sec, avg_tokens_per_sec, total_tokens_processed, elapsed_time, generated_text])
                    csv_file.flush()

                    save_checkpoint(opt_step, avg_micro_loss, val_loss)
                    model.train()

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C).")
        save_checkpoint(opt_step, emergency=True)
    except Exception as e:
        print(f"\n\nTraining crashed: {e}")
        save_checkpoint(opt_step, emergency=True)
        raise
    finally:
        pbar.close()
        csv_file.close()

    print(f"\nTraining complete. Metrics saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--config", type=str, default="default", choices=["default", "v2.1"],
                        help="Model config: 'default' (95M) or 'v2.1' (183M)")
    args = parser.parse_args()
    train(resume=args.resume, config_name=args.config)
