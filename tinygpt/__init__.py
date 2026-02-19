# tinygpt/__init__.py
"""TinyGPT: Educational and production-ready GPT implementation."""

__version__ = "1.0.0"

from .model import GPTLanguageModel, MoEGPTLanguageModel, WikipediaMoEGPTLanguageModel, TinyGPT2
from .config import GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig, TinyGPT2Config
from .tokenizer import Tokenizer
from .utils import generate

__all__ = [
    "GPTLanguageModel",
    "MoEGPTLanguageModel",
    "WikipediaMoEGPTLanguageModel",
    "TinyGPT2",
    "GPTConfig",
    "MoEGPTConfig",
    "WikipediaMoEGPTConfig",
    "TinyGPT2Config",
    "Tokenizer",
    "generate"
]