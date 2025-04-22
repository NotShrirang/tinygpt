import torch
import torch.nn.functional as F

from tinygpt.config import GPTConfig

def generate_square_subsequent_mask(sz):
    """
    Generates a causal (upper-triangular) mask for a sequence of length 'sz'.
    Positions with True (or -inf when using additive masks) will be masked.
    Here, we create an additive mask with -inf for masked positions.
    """
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.95):
    """
    Generate new tokens from the model given an initial sequence of indices 'idx'.
    The generation continues for 'max_new_tokens' steps.
    
    Args:
        model: The transformer model to use for generation
        idx: Initial token indices (batch_size, sequence_length)
        max_new_tokens: Number of new tokens to generate
        temperature: Controls randomness in sampling. Lower means more deterministic.
        top_k: If > 0, only sample from the top k most likely tokens
        top_p: If < 1.0, only sample from the smallest set of tokens whose cumulative probability exceeds p
    
    Yields:
        torch.Tensor: Each newly generated token as it's produced
    """
    model.eval()
    
    generated_tokens = idx.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = generated_tokens[:, -model.block_size:]
            logits, _ = model(seq)
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature

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
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

            yield next_token
