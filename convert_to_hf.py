"""
Convert TinyGPT2 checkpoints (.pth/.pt) to HuggingFace format.

Usage:
    python convert_to_hf.py \
        --checkpoint checkpoints_sft_alpace52k_openwebtext_95M/sft_epoch3.pth \
        --output_dir hf_model/ \
        --variant tinygpt2          # or tinygpt2.1
        --push_to_hub               # optional: push to HuggingFace Hub
        --hub_repo NotShrirang/TinyGPT2-SFT  # required if --push_to_hub
"""

import argparse
import json
import os
import shutil

import torch

from configuration_tinygpt2 import TinyGPT2HFConfig
from modeling_tinygpt2 import TinyGPT2ForCausalLM


VARIANT_DEFAULTS = {
    "tinygpt2": dict(
        n_embd=768, n_head=12, n_layer=12, gqa_kv_head=4, hidden_size=2048, dropout=0.1,
    ),
    "tinygpt2.1": dict(
        n_embd=1024, n_head=16, n_layer=12, gqa_kv_head=4, hidden_size=4096, dropout=0.1,
    ),
}


def load_original_state_dict(checkpoint_path):
    """Load checkpoint and extract model state dict, handling different save formats."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Strip _orig_mod. prefix from torch.compile()
    cleaned = {}
    for k, v in state_dict.items():
        k = k.removeprefix("_orig_mod.")
        cleaned[k] = v
    return cleaned


def convert(checkpoint_path, output_dir, variant="tinygpt2"):
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = load_original_state_dict(checkpoint_path)

    # Build HF config
    params = VARIANT_DEFAULTS[variant]
    config = TinyGPT2HFConfig(**params)

    # Instantiate HF model and load weights
    model = TinyGPT2ForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # freqs_cis is a buffer, not a parameter — expected to be "unexpected" or "missing"
    missing = [k for k in missing if "freqs_cis" not in k]
    unexpected = [k for k in unexpected if "freqs_cis" not in k]

    if missing:
        print(f"WARNING - missing keys: {missing}")
    if unexpected:
        print(f"WARNING - unexpected keys: {unexpected}")

    os.makedirs(output_dir, exist_ok=True)

    # Set auto_map so AutoModel/AutoConfig find our custom classes
    config.auto_map = {
        "AutoConfig": "configuration_tinygpt2.TinyGPT2HFConfig",
        "AutoModelForCausalLM": "modeling_tinygpt2.TinyGPT2ForCausalLM",
    }

    # Save model + config (handles tied weights correctly)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)  # overwrite with auto_map included

    # Copy the two custom code files so HF hub can use them
    for fname in ("configuration_tinygpt2.py", "modeling_tinygpt2.py"):
        src = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))

    print(f"HuggingFace model saved to: {output_dir}")
    print(f"Files: {os.listdir(output_dir)}")
    return output_dir


def push(output_dir, repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=repo_id)
    print(f"Pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert TinyGPT2 checkpoint to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth or .pt checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for HF model")
    parser.add_argument("--variant", default="tinygpt2", choices=list(VARIANT_DEFAULTS.keys()))
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", default=None, help="HuggingFace repo id (e.g. NotShrirang/TinyGPT2-SFT)")
    args = parser.parse_args()

    convert(args.checkpoint, args.output_dir, args.variant)

    if args.push_to_hub:
        if not args.hub_repo:
            parser.error("--hub_repo is required when using --push_to_hub")
        push(args.output_dir, args.hub_repo)


if __name__ == "__main__":
    main()
