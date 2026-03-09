import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import math
import time
import csv
import inspect
import random
import tqdm
from datasets import load_dataset
from tinygpt import TinyGPT2, TinyGPT2Config, TinyGPT2_1Config, Tokenizer


# Prompt template (same as SFT)
def format_prompt(instruction, system=""):
    parts = []
    if system.strip():
        parts.append(f"### System:\n{system}")
    parts.append(f"### Instruction:\n{instruction}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


# Random evaluation prompts
EVAL_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of regular exercise?",
    "Summarize the theory of relativity.",
    "How does a computer work?",
    "What is the difference between a virus and a bacteria?",
    "Explain what machine learning is.",
    "What causes earthquakes?",
    "How do airplanes fly?",
]


def get_model_config(config_name):
    if config_name == "v2.1":
        return TinyGPT2_1Config()
    return TinyGPT2Config()


def parse_hh_rlhf_dialogue(text):
    """Parse HH-RLHF dialogue format into prompt and final response.

    Format: '\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ...'
    Returns (prompt_text, final_assistant_response).
    """
    # Split on turn markers
    parts = text.split("\n\nAssistant:")
    if len(parts) < 2:
        return None, None

    # Everything before the last Assistant turn is the prompt
    prompt_context = "\n\nAssistant:".join(parts[:-1])
    final_response = parts[-1].strip()

    # Extract the human question from the prompt context
    human_parts = prompt_context.split("\n\nHuman:")
    if len(human_parts) < 2:
        return None, None

    # Use the last human message as instruction
    last_human = human_parts[-1].strip()

    return last_human, final_response


class HHRLHFDPODataset(Dataset):
    """Dataset for DPO training using Anthropic's HH-RLHF data."""

    def __init__(self, data, tokenizer, block_size):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.pad_id
        skipped = 0

        for item in data:
            chosen_text = item['chosen']
            rejected_text = item['rejected']

            # Parse dialogues
            instruction, chosen_response = parse_hh_rlhf_dialogue(chosen_text)
            _, rejected_response = parse_hh_rlhf_dialogue(rejected_text)

            if not instruction or not chosen_response or not rejected_response:
                skipped += 1
                continue

            # Skip identical pairs
            if chosen_response == rejected_response:
                skipped += 1
                continue

            # Format prompt in our template
            prompt = format_prompt(instruction)
            prompt_tokens = tokenizer.encode(prompt)

            # Tokenize chosen and rejected responses (append EOS token directly)
            chosen_tokens = tokenizer.encode(chosen_response) + [tokenizer.eos_id]
            rejected_tokens = tokenizer.encode(rejected_response) + [tokenizer.eos_id]

            # Build full sequences
            chosen_full = prompt_tokens + chosen_tokens
            rejected_full = prompt_tokens + rejected_tokens

            # Truncate if needed
            if len(chosen_full) > block_size:
                chosen_full = chosen_full[:block_size]
                chosen_tokens = chosen_full[len(prompt_tokens):]
            if len(rejected_full) > block_size:
                rejected_full = rejected_full[:block_size]
                rejected_tokens = rejected_full[len(prompt_tokens):]

            # Skip if response is too short after truncation
            if len(chosen_tokens) < 2 or len(rejected_tokens) < 2:
                skipped += 1
                continue

            # Create input/label pairs (shifted for next-token prediction)
            # Labels: -100 for prompt tokens, actual IDs for response tokens
            chosen_labels = [-100] * len(prompt_tokens) + chosen_tokens
            rejected_labels = [-100] * len(prompt_tokens) + rejected_tokens

            chosen_input = chosen_full[:-1]
            chosen_target = chosen_labels[1:]
            rejected_input = rejected_full[:-1]
            rejected_target = rejected_labels[1:]

            # Pad
            def pad_seq(seq, pad_val, length):
                pad_len = length - len(seq)
                if pad_len > 0:
                    return seq + [pad_val] * pad_len
                return seq[:length]

            chosen_input = pad_seq(chosen_input, self.pad_id, block_size)
            chosen_target = pad_seq(chosen_target, -100, block_size)
            rejected_input = pad_seq(rejected_input, self.pad_id, block_size)
            rejected_target = pad_seq(rejected_target, -100, block_size)

            self.examples.append({
                'chosen_input_ids': torch.tensor(chosen_input, dtype=torch.long),
                'chosen_labels': torch.tensor(chosen_target, dtype=torch.long),
                'rejected_input_ids': torch.tensor(rejected_input, dtype=torch.long),
                'rejected_labels': torch.tensor(rejected_target, dtype=torch.long),
            })

        if skipped > 0:
            print(f"  Skipped {skipped} examples (unparseable or identical)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_log_probs(logits, labels):
    """Compute per-token log probabilities for response tokens only.

    Args:
        logits: [B, T, V] model output logits
        labels: [B, T] target labels (-100 for prompt/padding tokens)

    Returns:
        Sum of log probs over response tokens for each example in batch [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log probs for the target tokens
    # labels shape: [B, T], log_probs shape: [B, T, V]
    target_log_probs = log_probs.gather(dim=-1, index=labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    # Mask: only count response tokens (where labels != -100)
    mask = (labels != -100).float()
    target_log_probs = target_log_probs * mask

    # Sum log probs per example
    return target_log_probs.sum(dim=-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Compute DPO loss.

    L_DPO = -log(sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x))))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    # Metrics
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
    chosen_reward_mean = chosen_rewards.mean()
    rejected_reward_mean = rejected_rewards.mean()

    return loss, reward_accuracies, chosen_reward_mean, rejected_reward_mean


def load_sft_model(checkpoint_path, device, config_name="default"):
    """Load an SFT checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = get_model_config(config_name)

    saved_config = checkpoint.get('config', {})
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)

    tokenizer = Tokenizer()
    model = TinyGPT2(config, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    n_params = sum(p.numel() for p in model.parameters())
    step = checkpoint.get('global_step', checkpoint.get('opt_step', '?'))

    return model, tokenizer, config, {
        'params': n_params, 'step': step,
        'train_loss': checkpoint.get('train_loss', '?'),
        'val_loss': checkpoint.get('val_loss', '?'),
    }


def print_banner(args, config, info, train_size, val_size, total_steps):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A"
    tokens_per_step = args.batch_size * args.gradient_accumulation_steps * config.block_size

    print(f"\n{'='*60}")
    print(f"  TinyGPT — Direct Preference Optimization (DPO)")
    print(f"  Model: TinyGPT2 {'v2.1 (~315M)' if args.config == 'v2.1' else 'default (~95M)'}")
    print(f"{'='*60}")
    print(f"  Device:         {device_name} ({vram} VRAM)")
    print(f"\n  SFT Checkpoint:")
    print(f"    Path:         {os.path.basename(args.checkpoint)}")
    print(f"    Step:         {info['step']}")
    print(f"    Train Loss:   {info['train_loss']:.4f}" if isinstance(info['train_loss'], float) else f"    Train Loss:   {info['train_loss']}")
    print(f"    Val Loss:     {info['val_loss']:.4f}" if isinstance(info['val_loss'], float) else f"    Val Loss:     {info['val_loss']}")
    print(f"    Parameters:   {info['params'] / 1e6:.2f}M")
    print(f"\n  Dataset:        Anthropic/hh-rlhf")
    print(f"    Train:        {train_size:,} preference pairs")
    print(f"    Val:          {val_size:,} preference pairs")
    print(f"\n  DPO Config:")
    print(f"    Epochs:       {args.epochs}")
    print(f"    Beta:         {args.beta}")
    print(f"    Batch size:   {args.batch_size} x {args.gradient_accumulation_steps} accum = {args.batch_size * args.gradient_accumulation_steps} effective")
    print(f"    Tokens/step:  {tokens_per_step:,}")
    print(f"    LR:           {args.lr}")
    print(f"    Total steps:  {total_steps:,}")
    print(f"    Eval every:   {args.eval_interval} steps")
    print(f"    Context:      {config.block_size} tokens")
    print(f"    Checkpoint:   checkpoints_dpo/")
    print(f"{'='*60}\n")


def evaluate_dpo(policy_model, ref_model, val_loader, device, dtype, beta):
    """Run DPO evaluation on validation set."""
    policy_model.eval()
    total_loss = 0
    total_accuracy = 0
    count = 0

    with torch.inference_mode():
        for batch in val_loader:
            chosen_ids = batch['chosen_input_ids'].to(device)
            chosen_labels = batch['chosen_labels'].to(device)
            rejected_ids = batch['rejected_input_ids'].to(device)
            rejected_labels = batch['rejected_labels'].to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                # Policy forward
                policy_chosen_logits, _, _ = policy_model(chosen_ids)
                policy_rejected_logits, _, _ = policy_model(rejected_ids)
                policy_chosen_logps = compute_log_probs(policy_chosen_logits, chosen_labels)
                policy_rejected_logps = compute_log_probs(policy_rejected_logits, rejected_labels)

                # Reference forward
                ref_chosen_logits, _, _ = ref_model(chosen_ids)
                ref_rejected_logits, _, _ = ref_model(rejected_ids)
                ref_chosen_logps = compute_log_probs(ref_chosen_logits, chosen_labels)
                ref_rejected_logps = compute_log_probs(ref_rejected_logits, rejected_labels)

                loss, accuracy, _, _ = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=beta
                )

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            count += 1

    policy_model.train()
    avg_loss = total_loss / max(count, 1)
    avg_accuracy = total_accuracy / max(count, 1)
    return avg_loss, avg_accuracy


def generate_sample(model, tokenizer, device, prompt_text=None):
    """Generate a sample response for evaluation."""
    if prompt_text is None:
        prompt_text = random.choice(EVAL_PROMPTS)
    prompt = format_prompt(prompt_text)
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    print(f"\nGenerated: {prompt}", end="", flush=True)
    with torch.inference_mode():
        generated = model.generate(
            prompt_tensor, max_new_tokens=100, temperature=0.7,
            top_k=40, tokenizer=tokenizer, stream=True,
            eos_token_id=tokenizer.eos_id,
        )
    generated_text = tokenizer.decode(generated[0].tolist())
    print()
    return generated_text


def train(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SFT model as policy
    print("Loading SFT checkpoint as policy model...")
    policy_model, tokenizer, config, info = load_sft_model(args.checkpoint, device, args.config)

    # Load SFT model as reference (frozen)
    print("Loading SFT checkpoint as reference model (frozen)...")
    ref_model, _, _, _ = load_sft_model(args.checkpoint, device, args.config)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    # Cast reference model to bf16 to save memory
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ref_model = ref_model.to(dtype)

    print(f"Policy model: {info['params'] / 1e6:.2f}M params (trainable)")
    print(f"Reference model: {info['params'] / 1e6:.2f}M params (frozen, {dtype})")

    # Load HH-RLHF dataset
    print("Loading Anthropic/hh-rlhf dataset...")
    ds = load_dataset("Anthropic/hh-rlhf")
    train_data = ds['train']
    val_data = ds['test']

    print(f"Processing {len(train_data):,} train + {len(val_data):,} test preference pairs...")
    train_dataset = HHRLHFDPODataset(train_data, tokenizer, config.block_size)
    val_dataset = HHRLHFDPODataset(val_data, tokenizer, config.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.epochs

    print_banner(args, config, info, len(train_dataset), len(val_dataset), total_steps)

    # Optimizer — very low LR for DPO
    optimizer = optim.AdamW(
        policy_model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Resume DPO checkpoint if requested
    start_epoch = 0
    global_step = 0
    total_tokens_processed = 0
    dpo_dir = "checkpoints_dpo"
    os.makedirs(dpo_dir, exist_ok=True)

    if args.resume:
        dpo_ckpts = sorted([f for f in os.listdir(dpo_dir) if f.endswith('.pth')],
                           key=lambda f: os.path.getmtime(os.path.join(dpo_dir, f)))
        if dpo_ckpts:
            latest = os.path.join(dpo_dir, dpo_ckpts[-1])
            print(f"Resuming DPO from {latest}")
            dpo_checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            policy_model.load_state_dict(dpo_checkpoint['model_state_dict'])
            optimizer.load_state_dict(dpo_checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(dpo_checkpoint['scheduler_state_dict'])
            start_epoch = dpo_checkpoint.get('epoch', 0)
            global_step = dpo_checkpoint.get('global_step', 0)
            total_tokens_processed = dpo_checkpoint.get('total_tokens_processed', 0)
            del dpo_checkpoint
            torch.cuda.empty_cache()
            print(f"Resumed at epoch {start_epoch}, step {global_step}, {total_tokens_processed:,} tokens processed")
        else:
            print("--resume passed but no DPO checkpoint found. Starting from scratch.")

    # CSV logging
    csv_path = os.path.join(dpo_dir, "dpo_metrics.csv")
    csv_mode = 'a' if args.resume and os.path.exists(csv_path) else 'w'
    csv_file = open(csv_path, csv_mode, newline='')
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    if csv_mode == 'w':
        csv_writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'train_reward_acc', 'val_reward_acc',
                             'chosen_reward', 'rejected_reward', 'learning_rate',
                             'tokens_per_sec', 'total_tokens_processed', 'elapsed_time', 'generated_text'])

    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    policy_model.train()
    start_time = time.time()

    print(f"Starting DPO training...")
    print(f"Epochs: {args.epochs}, Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}\n")

    def save_emergency_checkpoint():
        """Save emergency checkpoint on crash or Ctrl+C."""
        try:
            emergency_path = os.path.join(dpo_dir, "dpo_emergency.pth")
            print(f"\n{'!'*60}")
            print(f"  SAVING EMERGENCY CHECKPOINT at step {global_step}...")
            torch.save({
                'epoch': epoch if 'epoch' in dir() else start_epoch,
                'global_step': global_step,
                'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in policy_model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': running_loss / max(micro_step % args.gradient_accumulation_steps, 1),
                'total_tokens_processed': total_tokens_processed,
                'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')},
            }, emergency_path)
            print(f"  Emergency checkpoint saved to {emergency_path}")
            print(f"  Resume with: python dpo.py --checkpoint {args.checkpoint} --resume")
            print(f"{'!'*60}")
        except Exception as e:
            print(f"  Failed to save emergency checkpoint: {e}")

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"{'='*60}")
            print(f"  Epoch {epoch+1}/{args.epochs} ({steps_per_epoch} steps)")
            print(f"{'='*60}")

            # Calculate how many micro-batches to skip on resume
            skip_batches = global_step * args.gradient_accumulation_steps if epoch == start_epoch and global_step > 0 else 0
            resume_epoch_step = global_step if epoch == start_epoch and global_step > 0 else 0

            epoch_pbar = tqdm.tqdm(
                total=steps_per_epoch,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                initial=resume_epoch_step
            )

            running_loss = 0.0
            running_reward_acc = 0.0
            epoch_loss = 0.0
            micro_step = 0
            epoch_step = resume_epoch_step
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                # Skip already-processed batches on resume
                if batch_idx < skip_batches:
                    if batch_idx % 1000 == 0 and batch_idx > 0:
                        print(f"  Skipping batch {batch_idx}/{skip_batches}...")
                    continue

                t0 = time.time()

                chosen_ids = batch['chosen_input_ids'].to(device)
                chosen_labels = batch['chosen_labels'].to(device)
                rejected_ids = batch['rejected_input_ids'].to(device)
                rejected_labels = batch['rejected_labels'].to(device)

                num_tokens = (chosen_ids.shape[0] + rejected_ids.shape[0]) * chosen_ids.shape[1]
                total_tokens_processed += num_tokens

                with torch.amp.autocast('cuda', dtype=dtype):
                    # Policy forward on both chosen and rejected
                    policy_chosen_logits, _, _ = policy_model(chosen_ids)
                    policy_rejected_logits, _, _ = policy_model(rejected_ids)
                    policy_chosen_logps = compute_log_probs(policy_chosen_logits, chosen_labels)
                    policy_rejected_logps = compute_log_probs(policy_rejected_logits, rejected_labels)

                    # Reference forward (no grad)
                    with torch.no_grad():
                        ref_chosen_logits, _, _ = ref_model(chosen_ids)
                        ref_rejected_logits, _, _ = ref_model(rejected_ids)
                        ref_chosen_logps = compute_log_probs(ref_chosen_logits, chosen_labels)
                        ref_rejected_logps = compute_log_probs(ref_rejected_logits, rejected_labels)

                    loss, reward_acc, chosen_reward, rejected_reward = dpo_loss(
                        policy_chosen_logps, policy_rejected_logps,
                        ref_chosen_logps, ref_rejected_logps, beta=args.beta
                    )
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                running_loss += loss.item() * args.gradient_accumulation_steps
                running_reward_acc += reward_acc.item()
                micro_step += 1

                if micro_step % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    epoch_step += 1
                    epoch_pbar.update(1)

                    avg_loss = running_loss / args.gradient_accumulation_steps
                    avg_reward_acc = running_reward_acc / args.gradient_accumulation_steps
                    epoch_loss += avg_loss
                    running_loss = 0.0
                    running_reward_acc = 0.0

                    t1 = time.time()
                    tokens_per_sec = (num_tokens * args.gradient_accumulation_steps) / max(t1 - t0, 1e-6)
                    epoch_pbar.set_description(
                        f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Reward Acc: {avg_reward_acc:.2%} | {tokens_per_sec:.0f} tok/s"
                    )

                    # --- Evaluation & checkpoint ---
                    if global_step % args.eval_interval == 0:
                        val_loss, val_reward_acc = evaluate_dpo(
                            policy_model, ref_model, val_loader, device, dtype, args.beta
                        )

                        elapsed_time = time.time() - start_time
                        current_lr = optimizer.param_groups[0]['lr']

                        print(f"\n[Epoch {epoch+1}/{args.epochs} | Step {global_step}/{total_steps}]")
                        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                        print(f"Train Reward Acc: {avg_reward_acc:.2%}, Val Reward Acc: {val_reward_acc:.2%}")
                        print(f"Chosen Reward: {chosen_reward.item():.4f}, Rejected Reward: {rejected_reward.item():.4f}")
                        print(f"Elapsed: {elapsed_time:.1f}s, Throughput: {tokens_per_sec:.0f} tok/s")
                        print(f"Total tokens: {total_tokens_processed:,}")

                        # Generate sample
                        generated_text = generate_sample(policy_model, tokenizer, device)

                        csv_writer.writerow([epoch+1, global_step, avg_loss, val_loss, avg_reward_acc, val_reward_acc,
                                            chosen_reward.item(), rejected_reward.item(), current_lr,
                                            tokens_per_sec, total_tokens_processed, elapsed_time, generated_text])
                        csv_file.flush()

                        # Save checkpoint
                        ckpt_path = os.path.join(dpo_dir, f"dpo_step{global_step}.pth")
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in policy_model.state_dict().items()},
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': avg_loss,
                            'val_loss': val_loss,
                            'reward_accuracy': val_reward_acc,
                            'total_tokens_processed': total_tokens_processed,
                            'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')},
                        }, ckpt_path)
                        print(f"Checkpoint saved to {ckpt_path}")

                        policy_model.train()

            epoch_pbar.close()

            # End of epoch evaluation
            avg_epoch_loss = epoch_loss / max(epoch_step, 1)
            val_loss, val_reward_acc = evaluate_dpo(
                policy_model, ref_model, val_loader, device, dtype, args.beta
            )
            elapsed_time = time.time() - start_time

            print(f"\n{'─'*60}")
            print(f"  Epoch {epoch+1}/{args.epochs} complete")
            print(f"  Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Val Reward Accuracy: {val_reward_acc:.2%}")
            print(f"  Elapsed: {elapsed_time:.0f}s | Total tokens: {total_tokens_processed:,}")

            generated_text = generate_sample(policy_model, tokenizer, device)
            print(f"{'─'*60}\n")

            # Save end-of-epoch checkpoint
            ckpt_path = os.path.join(dpo_dir, f"dpo_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': {k.replace("_orig_mod.", ""): v for k, v in policy_model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_epoch_loss,
                'val_loss': val_loss,
                'reward_accuracy': val_reward_acc,
                'total_tokens_processed': total_tokens_processed,
                'config': {k: v for k, v in inspect.getmembers(config) if not k.startswith('__')},
            }, ckpt_path)
            print(f"  Epoch checkpoint saved to {ckpt_path}\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)!")
        save_emergency_checkpoint()
    except Exception as e:
        print(f"\n\nTraining crashed with error: {e}")
        save_emergency_checkpoint()
        raise
    finally:
        csv_file.close()

    print(f"DPO training complete! Metrics saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyGPT DPO Training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume DPO from latest DPO checkpoint")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (default: 5e-7)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps (default: 16)")
    parser.add_argument("--eval_interval", type=int, default=200, help="Evaluate every N optimizer steps (default: 200)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto)")
    parser.add_argument("--config", type=str, default="default", choices=["default", "v2.1"],
                        help="Model config: 'default' (95M) or 'v2.1' (315M)")
    args = parser.parse_args()

    train(args)
