import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Check if it is linux
if os.name == 'posix' and torch.cuda.is_available():
    from liger_kernel.transformers import LigerSwiGLUMLP, liger_rotary_pos_emb, LigerFusedLinearCrossEntropyLoss
else:
    LigerFusedLinearCrossEntropyLoss = None

from tinygpt.config import GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig, TinyGPT2Config
from tinygpt.layers import DecoderBlock, CausalMoEBlock, RotaryEmbeddings, WikipediaCausalMoEBlock, TinyGPT2Block, get_rms_norm, precompute_freqs_cis
from tinygpt.utils import remove_orig_mod_prefix, map_swiglu_keys


class GPTLanguageModel(nn.Module):
    """
    A simple GPT language model with a stack of transformer blocks.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)

        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(self.n_layer)])
        
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        self.apply(self._init_weights)

        self.token_embedding_table.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    @classmethod
    def from_pretrained(self, pretrained_model_path: str, device: str = "cpu") -> "GPTLanguageModel":
        """
        Load a pretrained model from the specified path.
        """
        model = self(GPTConfig())
        state_dict = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Given a sequence of indices 'idx', generate 'max_new_tokens' new tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class MoEGPTLanguageModel(nn.Module):
    def __init__(self, config: MoEGPTConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.ModuleList([CausalMoEBlock(self.config, self.device) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.block_size = self.config.block_size

        if self.config.pad_token_id:
            self.loss_fct = LigerFusedLinearCrossEntropyLoss(ignore_index=self.config.pad_token_id).to(self.lm_head.weight.device)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, inference=False):
        B,T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(
            torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if inference or targets is None:
            return logits if inference else (logits, None)
        logits_flat = logits.view(-1, self.config.vocab_size)
        target_flat = targets.view(-1)
        if hasattr(self, 'loss_fct'):
            loss = self.loss_fct(self.lm_head.weight,
                                 x.view(-1, self.config.n_embd),
                                 target_flat)
        else:
            loss = F.cross_entropy(logits_flat, target_flat, ignore_index=self.config.pad_token_id)
        return logits, loss
    
    @classmethod
    def from_pretrained(self, pretrained_model_path: str, device: str = "cpu") -> "MoEGPTLanguageModel":
        """
        Load a pretrained model from the specified path.
        """
        model = self(MoEGPTConfig(), device=device)
        state_dict = torch.load(pretrained_model_path, map_location=device)
        state_dict = remove_orig_mod_prefix(state_dict)
        state_dict = map_swiglu_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")

        model.eval()
        model.to(device)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits,_ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class WikipediaMoEGPTLanguageModel(nn.Module):
    """
    Wikipedia-trained Mixture of Experts GPT Language Model with 8 experts.
    Based on the architecture from wikipedia_titoken_MoE_train.py
    """
    def __init__(self, config: WikipediaMoEGPTConfig, device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd, device=device)
        self.blocks = nn.ModuleList([WikipediaCausalMoEBlock(config, device) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device)
        self.lm_head.weight = self.token_emb.weight
        self.block_size = config.block_size

        if config.pad_token_id and LigerFusedLinearCrossEntropyLoss:
            self.loss_fct = LigerFusedLinearCrossEntropyLoss(ignore_index=config.pad_token_id).to(device)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, inference=False):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if inference or targets is None:
            return logits if inference else (logits, None)
        
        logits_flat = logits.view(-1, self.config.vocab_size)
        target_flat = targets.view(-1)
        
        if hasattr(self, 'loss_fct'):
            loss = self.loss_fct(self.lm_head.weight,
                                 x.view(-1, self.config.n_embd),
                                 target_flat)
        else:
            loss = F.cross_entropy(logits_flat, target_flat, ignore_index=self.config.pad_token_id)
        
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str = "cpu") -> "WikipediaMoEGPTLanguageModel":
        """
        Load a pretrained Wikipedia MoE model from the specified path.
        """
        config = WikipediaMoEGPTConfig()
        model = cls(config, device=device)
        state_dict = torch.load(pretrained_model_path, map_location=device)
        state_dict = remove_orig_mod_prefix(state_dict)
        state_dict = map_swiglu_keys(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys[:10]}...")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys[:10]}...")

        model.eval()
        model.to(device)
        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens using the Wikipedia MoE model.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class TinyGPT2(nn.Module):
    def __init__(self, config: TinyGPT2Config, pad_id: int = 50257):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.ModuleList([TinyGPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = get_rms_norm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight  # Weight tying

        # Precompute RoPE frequencies (*2 for extrapolation safety)
        self.register_buffer('freqs_cis', precompute_freqs_cis(config.n_embd // config.n_head, config.block_size * 2))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None, start_pos=None):
        B, T = idx.shape

        x = self.token_embedding(idx)

        if kv_caches is not None and start_pos is not None:
            freqs_cis = self.freqs_cis[start_pos:start_pos + T]
        else:
            freqs_cis = self.freqs_cis[:T]

        is_causal = True

        new_kv_caches = []

        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, freqs_cis, is_causal=is_causal, kv_cache=kv_cache)
            new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, self.config.vocab_size)
            targets = targets.view(-1)
            loss = self.loss_fn(logits, targets)

        return logits, loss, new_kv_caches

    @torch.inference_mode()
    @torch.compiler.disable
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, tokenizer=None, stream=False, eos_token_id=None):
        self.eval()
        B, T = idx.shape
        eos_text = "<|endoftext|>"
        # Buffer recent tokens to detect literal <|endoftext|> and avoid printing it
        stream_buffer = []
        eos_text_len = len(eos_text)

        def _flush_buffer(force=False):
            """Flush buffered tokens, holding back enough to detect EOS text."""
            if not stream or not tokenizer:
                return
            if force:
                text = tokenizer.decode(stream_buffer)
                if eos_text in text:
                    text = text[:text.index(eos_text)]
                print(text, end="", flush=True)
                stream_buffer.clear()
            elif len(stream_buffer) > eos_text_len:
                # Decode all buffered tokens, print all but the last eos_text_len chars
                text = tokenizer.decode(stream_buffer)
                if len(text) > eos_text_len:
                    safe = text[:-eos_text_len]
                    print(safe, end="", flush=True)
                    # Re-encode the held-back portion to keep buffer accurate
                    remaining = tokenizer.encode(text[-eos_text_len:])
                    stream_buffer.clear()
                    stream_buffer.extend(remaining)

        # Initial prefill pass
        logits, _, kv_caches = self(idx, kv_caches=None, start_pos=None)
        cur_pos = T

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if eos_token_id is not None and idx_next.item() == eos_token_id:
            return idx

        stream_buffer.append(idx_next.item())
        _flush_buffer()

        # Autoregressive decoding with KV cache
        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self(idx_next, kv_caches=kv_caches, start_pos=cur_pos)
            cur_pos += 1
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            # Check for EOS token ID
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break

            stream_buffer.append(idx_next.item())

            # Check for literal <|endoftext|> in recent decoded text
            if eos_token_id is not None and tokenizer:
                tail = tokenizer.decode(stream_buffer[-10:])
                if eos_text in tail:
                    break

            _flush_buffer()

        # Flush any remaining buffered tokens
        _flush_buffer(force=True)

        return idx

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, device: str = "cpu") -> "TinyGPT2":
        from tinygpt.tokenizer import Tokenizer

        config = TinyGPT2Config()
        tokenizer = Tokenizer()
        model = cls(config, pad_id=tokenizer.pad_id)

        state_dict = torch.load(pretrained_model_path, map_location=device, weights_only=False)

        # Handle checkpoint format (dict with 'model_state_dict' key)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        state_dict = remove_orig_mod_prefix(state_dict)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        return model
