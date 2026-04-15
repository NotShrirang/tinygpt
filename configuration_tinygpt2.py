"""HuggingFace-compatible configuration for TinyGPT2 models."""

from transformers import PretrainedConfig


class TinyGPT2HFConfig(PretrainedConfig):
    model_type = "tinygpt2"

    def __init__(
        self,
        vocab_size=50304,
        block_size=512,
        n_embd=768,
        n_head=12,
        n_layer=12,
        gqa_kv_head=4,
        hidden_size=2048,
        dropout=0.1,
        pad_token_id=50257,
        eos_token_id=50256,
        bos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.gqa_kv_head = gqa_kv_head
        self.hidden_size = hidden_size
        self.dropout = dropout
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs,
        )
