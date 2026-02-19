import torch
import argparse
import os
from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    saved_config = checkpoint['config']
    config = TinyGPT2Config()
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    config.device = device

    # Build model and load weights
    tokenizer = Tokenizer()
    model = TinyGPT2(config, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    step = checkpoint.get('opt_step', '?')
    train_loss = checkpoint.get('train_loss', '?')
    val_loss = checkpoint.get('val_loss', '?')
    tokens = checkpoint.get('total_tokens_processed', '?')

    print(f"{'='*50}")
    print(f"  TinyGPT Inference")
    print(f"{'='*50}")
    print(f"  Checkpoint:  {os.path.basename(checkpoint_path)}")
    print(f"  Step:        {step}")
    print(f"  Train Loss:  {train_loss}")
    print(f"  Val Loss:    {val_loss}")
    print(f"  Tokens seen: {tokens:,}" if isinstance(tokens, int) else f"  Tokens seen: {tokens}")
    print(f"  Parameters:  {n_params / 1e6:.2f}M")
    print(f"  Device:      {device}")
    print(f"{'='*50}\n")

    return model, tokenizer, config


def generate_text(model, tokenizer, config, prompt, max_tokens, temperature, top_k):
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(config.device)

    print(prompt, end="", flush=True)
    with torch.inference_mode():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
            stream=True,
        )
    print()

    return tokenizer.decode(generated[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="TinyGPT Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (if omitted, enters interactive mode)")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling (default: 40)")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config = load_model(args.checkpoint, device)

    if args.prompt:
        generate_text(model, tokenizer, config, args.prompt, args.max_tokens, args.temperature, args.top_k)
    else:
        print("Interactive mode — type your prompt and press Enter. Ctrl+C to exit.\n")
        try:
            while True:
                prompt = input(">>> ")
                if not prompt.strip():
                    continue
                generate_text(model, tokenizer, config, prompt, args.max_tokens, args.temperature, args.top_k)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


if __name__ == "__main__":
    main()
