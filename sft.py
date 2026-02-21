import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import math
import time
import csv
import inspect
import tqdm
from datasets import load_dataset
from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer


# SFT Prompt Template
def format_prompt(instruction, input_text=""):
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


class AlpacaSFTDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.pad_id

        for item in data:
            prompt = format_prompt(item['instruction'], item.get('input', ''))
            response = item['output'] + "<|endoftext|>"

            prompt_tokens = tokenizer.encode(prompt)
            response_tokens = tokenizer.encode(response)

            full_tokens = prompt_tokens + response_tokens
            if len(full_tokens) > block_size:
                # Truncate but keep at least some response
                max_prompt = block_size - min(len(response_tokens), block_size // 2)
                prompt_tokens = prompt_tokens[:max_prompt]
                remaining = block_size - len(prompt_tokens)
                response_tokens = response_tokens[:remaining]
                full_tokens = prompt_tokens + response_tokens

            # Create labels: -100 for prompt tokens (no loss), actual tokens for response
            labels = [-100] * len(prompt_tokens) + response_tokens

            # Shift for next-token prediction: input is full_tokens[:-1], labels is full_tokens[1:]
            input_ids = full_tokens[:-1]
            target_ids = labels[1:]

            # Pad to block_size
            pad_len = block_size - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_id] * pad_len
                target_ids = target_ids + [-100] * pad_len

            self.examples.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(target_ids, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_pretrained(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_config = checkpoint['config']
    config = TinyGPT2Config()
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.device = device

    tokenizer = Tokenizer()
    model = TinyGPT2(config, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    n_params = sum(p.numel() for p in model.parameters())
    step = checkpoint.get('opt_step', '?')
    train_loss = checkpoint.get('train_loss', '?')
    val_loss = checkpoint.get('val_loss', '?')

    return model, tokenizer, config, {
        'params': n_params, 'step': step,
        'train_loss': train_loss, 'val_loss': val_loss,
    }


def print_banner(args, config, info, train_size, val_size, total_steps):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * config.block_size

    print(f"\n{'='*60}")
    print(f"  TinyGPT — Supervised Fine-Tuning (SFT)")
    print(f"{'='*60}")
    print(f"  Device:         {device_name} ({vram} VRAM)")
    print(f"\n  Pretrained Checkpoint:")
    print(f"    Path:         {os.path.basename(args.checkpoint)}")
    print(f"    Step:         {info['step']}")
    print(f"    Train Loss:   {info['train_loss']:.4f}" if isinstance(info['train_loss'], float) else f"    Train Loss:   {info['train_loss']}")
    print(f"    Val Loss:     {info['val_loss']:.4f}" if isinstance(info['val_loss'], float) else f"    Val Loss:     {info['val_loss']}")
    print(f"    Parameters:   {info['params'] / 1e6:.2f}M")
    print(f"\n  Dataset:        Stanford Alpaca (52K)")
    print(f"    Train:        {train_size:,} examples")
    print(f"    Val:          {val_size:,} examples")
    print(f"\n  SFT Config:")
    print(f"    Epochs:       {args.epochs}")
    print(f"    Batch size:   {args.batch_size} x {args.gradient_accumulation_steps} accum = {args.batch_size * args.gradient_accumulation_steps} effective")
    print(f"    Tokens/step:  {tokens_per_step:,}")
    print(f"    LR:           {args.lr}")
    print(f"    Total steps:  {total_steps:,}")
    print(f"    Eval every:   {args.eval_interval} steps")
    print(f"    Context:      {config.block_size} tokens")
    print(f"    Checkpoint:   checkpoints_sft/")
    print(f"{'='*60}\n")


def evaluate(model, val_loader, device, dtype):
    model.eval()
    total_loss = 0
    count = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.inference_mode():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                logits, _, _ = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / count if count > 0 else float('inf')


def generate_sample(model, tokenizer, device, prompt_text="What is the capital of France?"):
    prompt = format_prompt(prompt_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    print(f"\nGenerated: {prompt}", end="", flush=True)
    with torch.inference_mode():
        generated = model.generate(
            prompt_tensor, max_new_tokens=100, temperature=0.7,
            top_k=40, tokenizer=tokenizer, stream=True
        )
    generated_text = tokenizer.decode(generated[0].tolist())
    print()
    return generated_text


def train(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained model
    model, tokenizer, config, info = load_pretrained(args.checkpoint, device)

    # Load Alpaca dataset
    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.train_test_split(test_size=0.1, seed=42)
    train_data = ds['train']
    val_data = ds['test']

    train_dataset = AlpacaSFTDataset(train_data, tokenizer, config.block_size)
    val_dataset = AlpacaSFTDataset(val_data, tokenizer, config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    print_banner(args, config, info, len(train_dataset), len(val_dataset), total_steps)

    # Loss function with -100 masking (ignores prompt tokens)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer — lower LR to avoid catastrophic forgetting
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    warmup_steps = min(100, total_steps // 10)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume SFT checkpoint if requested
    start_epoch = 0
    global_step = 0
    total_tokens_processed = 0
    sft_dir = "checkpoints_sft"
    os.makedirs(sft_dir, exist_ok=True)

    if args.resume:
        sft_ckpts = sorted([f for f in os.listdir(sft_dir) if f.endswith('.pth')])
        if sft_ckpts:
            latest = os.path.join(sft_dir, sft_ckpts[-1])
            print(f"Resuming SFT from {latest}")
            sft_checkpoint = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(sft_checkpoint['model_state_dict'])
            optimizer.load_state_dict(sft_checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(sft_checkpoint['scheduler_state_dict'])
            start_epoch = sft_checkpoint.get('epoch', 0)
            global_step = sft_checkpoint.get('global_step', 0)
            total_tokens_processed = sft_checkpoint.get('total_tokens_processed', 0)
            print(f"Resumed at epoch {start_epoch}, step {global_step}, {total_tokens_processed:,} tokens processed")
        else:
            print("--resume passed but no SFT checkpoint found. Starting from scratch.")

    # CSV logging
    csv_path = os.path.join(sft_dir, "sft_metrics.csv")
    csv_mode = 'a' if args.resume and os.path.exists(csv_path) else 'w'
    csv_file = open(csv_path, csv_mode, newline='')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    if csv_mode == 'w':
        csv_writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'train_perplexity', 'val_perplexity',
                             'learning_rate', 'tokens_per_sec', 'avg_tokens_per_sec', 'total_tokens_processed',
                             'elapsed_time', 'generated_text'])

    # Use mixed precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    model.train()
    start_time = time.time()

    print(f"Starting SFT training...")
    print(f"Epochs: {args.epochs}, Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"{'='*60}")
        print(f"  Epoch {epoch+1}/{args.epochs} ({steps_per_epoch} steps)")
        print(f"{'='*60}")

        # Progress bar per epoch
        epoch_pbar = tqdm.tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            initial=0
        )

        running_loss = 0.0
        epoch_loss = 0.0
        micro_step = 0
        epoch_step = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            num_tokens = input_ids.shape[0] * input_ids.shape[1]
            total_tokens_processed += num_tokens

            with torch.amp.autocast('cuda', dtype=dtype):
                logits, _, _ = model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss = loss_fn(logits, labels_flat)
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.gradient_accumulation_steps
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                epoch_step += 1
                epoch_pbar.update(1)

                avg_loss = running_loss / args.gradient_accumulation_steps
                epoch_loss += avg_loss
                running_loss = 0.0

                t1 = time.time()
                tokens_per_sec = (num_tokens * args.gradient_accumulation_steps) / max(t1 - t0, 1e-6)
                epoch_pbar.set_description(
                    f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | {tokens_per_sec:.0f} tok/s"
                )

                # --- Evaluation & checkpoint ---
                if global_step % args.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, dtype)

                    train_perplexity = math.exp(min(avg_loss, 20))
                    val_perplexity = math.exp(min(val_loss, 20))

                    elapsed_time = time.time() - start_time
                    avg_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
                    current_lr = optimizer.param_groups[0]['lr']

                    print(f"\n[Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps}] Train Loss {avg_loss:.4f}, Val Loss {val_loss:.4f}")
                    print(f"Train PPL: {train_perplexity:.2f}, Val PPL: {val_perplexity:.2f}")
                    print(f"Elapsed: {elapsed_time:.1f}s, Avg throughput: {avg_tokens_per_sec:.0f} tok/s")
                    print(f"Total tokens: {total_tokens_processed:,}")

                    # Generate sample text
                    generated_text = generate_sample(model, tokenizer, device)

                    csv_writer.writerow([epoch+1, global_step, avg_loss, val_loss, train_perplexity, val_perplexity,
                                        current_lr, tokens_per_sec, avg_tokens_per_sec, total_tokens_processed,
                                        elapsed_time, generated_text])
                    csv_file.flush()

                    # Save checkpoint
                    ckpt_path = os.path.join(sft_dir, f"sft_step{global_step}.pth")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': avg_loss,
                        'val_loss': val_loss,
                        'total_tokens_processed': total_tokens_processed,
                        'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')},
                    }, ckpt_path)
                    print(f"Checkpoint saved to {ckpt_path}")

                    model.train()

        epoch_pbar.close()

        # End of epoch evaluation
        avg_epoch_loss = epoch_loss / max(epoch_step, 1)
        val_loss = evaluate(model, val_loader, device, dtype)
        elapsed_time = time.time() - start_time

        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{args.epochs} complete")
        print(f"  Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train PPL: {math.exp(min(avg_epoch_loss, 20)):.2f} | Val PPL: {math.exp(min(val_loss, 20)):.2f}")
        print(f"  Elapsed: {elapsed_time:.0f}s | Total tokens: {total_tokens_processed:,}")

        generated_text = generate_sample(model, tokenizer, device)
        print(f"{'─'*60}\n")

        # Save end-of-epoch checkpoint
        ckpt_path = os.path.join(sft_dir, f"sft_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_epoch_loss,
            'val_loss': val_loss,
            'total_tokens_processed': total_tokens_processed,
            'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')},
        }, ckpt_path)
        print(f"  Epoch checkpoint saved to {ckpt_path}\n")

    csv_file.close()
    print(f"SFT training complete! Metrics saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyGPT SFT Training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume SFT from latest SFT checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate every N optimizer steps (default: 100)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    train(args)
