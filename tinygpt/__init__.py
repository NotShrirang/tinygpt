# tinygpt/__init__.py
"""TinyGPT: Educational and production-ready GPT implementation."""

__version__ = "1.0.0"

from .model import GPTLanguageModel, MoEGPTLanguageModel
from .config import GPTConfig, MoEGPTConfig
from .tokenizer import Tokenizer
from .utils import generate

__all__ = [
    "GPTLanguageModel",
    "MoEGPTLanguageModel", 
    "GPTConfig",
    "MoEGPTConfig",
    "Tokenizer",
    "generate"
]