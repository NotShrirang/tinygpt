from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.3

    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

@dataclass
class MoEGPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    dropout: float = 0.3

    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    n_experts: int = 4
    top_experts: int = 2
    noisy_topk: bool = False
    use_checkpointing: bool = False
    pad_token_id = None
