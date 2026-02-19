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

@dataclass
class WikipediaMoEGPTConfig:
    vocab_size: int = 50304
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 16
    n_layer: int = 8
    dropout: float = 0.3

    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95

    # Wikipedia MoE specific parameters
    n_experts: int = 8
    top_experts: int = 2
    noisy_topk: bool = False
    use_checkpointing: bool = False
    pad_token_id: int = 50257  # PAD token from tiktoken

@dataclass
class TinyGPT2Config:
    vocab_size: int = 50304
    block_size: int = 512
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    gqa_kv_head: int = 4
    hidden_size: int = 2048
    dropout: float = 0.1

    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
