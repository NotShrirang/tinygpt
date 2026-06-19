import inspect

import torch
import torch.nn.functional as F
from typing import Iterator, Optional

from tinygpt.config import GPTConfig

def generate_square_subsequent_mask(sz):
    """
    Generates a causal (upper-triangular) mask for a sequence of length 'sz'.
    Positions with True (or -inf when using additive masks) will be masked.
    Here, we create an additive mask with -inf for masked positions.
    """
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def generate_square_subsequent_mask_with_device(sz:int, device=None):
    return torch.triu(torch.full((sz,sz), float('-inf'), device=device if device else torch.device('cpu')), diagonal=1)

def _supports_kv_cache(model) -> bool:
    """True if the model's forward accepts a `kv_caches` argument (incremental decode)."""
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        return "kv_caches" in inspect.signature(forward).parameters
    except (ValueError, TypeError):
        return False


def _sample_next(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    word_repetition_penalty: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply repetition penalty + temperature + top-k/top-p, then sample one token.

    `logits` are the last-position logits (B, vocab); `generated_tokens` (B, T) is
    the full running sequence, used only for the repetition penalty.
    """
    if word_repetition_penalty != 1.0:
        for batch_idx in range(generated_tokens.shape[0]):
            for token_idx in torch.unique(generated_tokens[batch_idx]):
                if token_idx >= 0:
                    logits[batch_idx, token_idx] = logits[batch_idx, token_idx] / word_repetition_penalty

    if temperature > 0:
        logits = logits / (temperature + eps)

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for batch_idx in range(logits.shape[0]):
            indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
            logits[batch_idx, indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    word_repetition_penalty: float = 1.0
) -> Iterator[torch.Tensor]:
    """
    Generate new tokens from the model given an initial sequence of indices 'idx'.
    The generation continues for 'max_new_tokens' steps.

    Uses an incremental KV cache when the model supports it (forward accepts
    `kv_caches`), which makes each decode step O(1) in context length instead of
    re-running the whole sequence every step. Models without cache support (e.g.
    the TensorRT runner, older variants) fall back to the full-recompute path.

    Args:
        model: The transformer model to use for generation
        idx: Initial token indices (batch_size, sequence_length)
        max_new_tokens: Number of new tokens to generate
        temperature: Controls randomness in sampling. Lower means more deterministic.
        top_k: If > 0, only sample from the top k most likely tokens
        top_p: If < 1.0, only sample from the smallest set of tokens whose cumulative probability exceeds p
        word_repetition_penalty: Penalty for repeating words in the generated text

    Yields:
        torch.Tensor: Each newly generated token as it's produced
    """
    model.eval()
    if _supports_kv_cache(model):
        yield from _generate_kv_cached(
            model, idx, max_new_tokens, temperature, top_k, top_p, word_repetition_penalty
        )
    else:
        yield from _generate_full(
            model, idx, max_new_tokens, temperature, top_k, top_p, word_repetition_penalty
        )


def _generate_full(
    model, idx, max_new_tokens, temperature, top_k, top_p, word_repetition_penalty
) -> Iterator[torch.Tensor]:
    """Fallback decode: re-run the full (truncated) sequence every step. O(T) per token."""
    generated_tokens = idx.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = generated_tokens[:, -model.block_size:]
            result = model(seq)
            logits = result[0][:, -1, :]

            next_token = _sample_next(
                logits, generated_tokens, temperature, top_k, top_p, word_repetition_penalty
            )
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            yield next_token


def _generate_kv_cached(
    model, idx, max_new_tokens, temperature, top_k, top_p, word_repetition_penalty
) -> Iterator[torch.Tensor]:
    """Incremental decode with a KV cache: prefill once, then feed one token per step."""
    device = next(model.parameters()).device
    block_size = getattr(model, "block_size", idx.shape[1])
    # Cap the prompt to the context window so RoPE positions stay in range.
    idx = idx[:, -block_size:].to(device)
    generated_tokens = idx.clone()

    with torch.no_grad():
        logits, _, kv_caches = model(idx, kv_caches=None, start_pos=None)
        cur_pos = idx.shape[1]

        for step in range(max_new_tokens):
            last_logits = logits[:, -1, :]
            next_token = _sample_next(
                last_logits, generated_tokens, temperature, top_k, top_p, word_repetition_penalty
            )
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            yield next_token

            # Skip the extra forward after the final token.
            if step < max_new_tokens - 1:
                logits, _, kv_caches = model(next_token, kv_caches=kv_caches, start_pos=cur_pos)
                cur_pos += 1

def remove_orig_mod_prefix(state_dict):
    """
    Remove '_orig_mod.' prefix from state dict keys.
    This is needed when loading models that were saved with torch.compile() or similar optimizations.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def map_swiglu_keys(state_dict):
    """
    Map swiglu keys from liger_kernel format to fallback format.
    This handles the difference between:
    - Saved: blocks.X.moe.experts.Y.swiglu.gate_proj.weight
    - Expected: blocks.X.moe.experts.Y.gate_proj.weight
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if '.swiglu.' in key:
            # Remove the .swiglu. part from the key
            new_key = key.replace('.swiglu.', '.')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
