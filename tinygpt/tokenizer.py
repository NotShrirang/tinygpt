import tiktoken
from typing import List


class Tokenizer:
    def __init__(self, tokenizer_model="gpt2"):
        gpt2_enc = tiktoken.get_encoding(tokenizer_model)
        self.enc = tiktoken.Encoding(
            name=tokenizer_model,
            pat_str=gpt2_enc._pat_str,
            mergeable_ranks=gpt2_enc._mergeable_ranks,
			special_tokens={
                **gpt2_enc._special_tokens,
                "PAD": 50257,
			},
		)
        self.tokenizer_model = tokenizer_model

        self.n_words = self.enc.n_vocab
        self.bos_id = None
        self.eos_id = self.enc.eot_token
        self.pad_id = self.enc._special_tokens["PAD"]

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        t = self.enc.encode(s, disallowed_special=())
        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)
