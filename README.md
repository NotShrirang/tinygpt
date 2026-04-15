<img src="https://github.com/user-attachments/assets/8a90d976-57cb-4e3b-a9e7-33e37816eb81" alt="TinyGPT Banner" />

# TinyGPT 🤖

> **NEW: TinyGPT2-SFT is here!** 🎉 Our 95M parameter model is now instruction fine-tuned on Stanford Alpaca. It can follow instructions, answer questions, write poems, and more — all trained on a single RTX 3070 Ti. [Try it out!](https://tinygpt.streamlit.app/)

[![GitHub stars](https://img.shields.io/github/stars/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/network/members)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinygpt.streamlit.app/)

**TinyGPT** is an educational implementation of the GPT (Generative Pre-trained Transformer) architecture, featuring five model variants — from simple story generators to an instruction-following model built with RoPE, GQA, and RMSNorm. Built from the ground up with modern PyTorch, TinyGPT demonstrates how state-of-the-art language models can be both accessible and performant. ✨

🔗 **Quick Links:**

- 🤗 [HuggingFace Repository](https://huggingface.co/NotShrirang/tinygpt)
- 🤗 [TinyGPT2-IT (Instruction-Tuned)](https://huggingface.co/NotShrirang/tinygpt2-it)
- 🚀 [Live Demo](https://tinygpt.streamlit.app/)
- 📚 [Training Notebooks](./notebooks/)

## Overview 🔍

TinyGPT represents a carefully crafted balance between **accessibility** and **performance** in language model design. The project progresses through multiple model variants — from a standard GPT to Mixture-of-Experts architectures to instruction fine-tuned and preference-aligned TinyGPT2 models with cutting-edge techniques including DPO alignment.

### 🎯 **Project Goals**

- **Educational**: Provide a clear, well-documented implementation of GPT architecture
- **Production-Ready**: Deliver robust, efficient models suitable for real-world applications
- **Efficient**: Optimized for running on consumer GPUs and edge devices with minimal latency
- **Accessible**: Make it easy to run, train, fine-tune, and deploy on various platforms

## Model Architecture 🏗️

TinyGPT comes in five variants:

### TinyGPT (Standard) 🤖

<img width="1748" height="1240" alt="TinyGPT 51M" src="https://github.com/user-attachments/assets/984fabe1-e714-453a-958d-63138e34b941" />

- 8 transformer blocks 🧱
- 8 attention heads 👁️
- 512 embedding dimensions 📊
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~51M
- Training data: TinyStories dataset

### TinyGPT-MoE (Mixture of Experts) 🧠

<img width="1748" height="1240" alt="TinyGPT MoE 84M" src="https://github.com/user-attachments/assets/80e3b1d8-c027-41f3-ba5f-551557f65223" />

- 8 transformer blocks with MoE layers 🧱
- 8 attention heads 👁️
- 512 embedding dimensions 📊
- 4 experts per MoE layer with top-2 routing 🔀
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~85M
- Training data: TinyStories dataset
- Enhanced storytelling capabilities through expert specialization

### Wikipedia-MoE 🌐

- 8 transformer blocks with MoE layers 🧱
- 16 attention heads 👁️
- 512 embedding dimensions 📊
- 8 experts per MoE layer with top-2 routing 🔀
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~135M
- Training data: Wikipedia (C4 dataset)
- Enhanced knowledge representation with more experts and attention heads

### TinyGPT2 ⚡

- 12 transformer blocks 🧱
- 12 attention heads with Grouped Query Attention (4 KV groups) 👁️
- 768 embedding dimensions 📊
- 2048 FFN hidden size 🔧
- RoPE (Rotary Position Embeddings) for position encoding 🔄
- RMSNorm for layer normalization 📏
- KV Cache for efficient autoregressive generation 🚀
- Weight tying between token embeddings and output head 🔗
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~95M
- Training data: OpenWebText (~6.5B+ tokens)

### TinyGPT2.1 ⚡⚡

- 12 transformer blocks 🧱
- 16 attention heads with Grouped Query Attention (4 KV groups) 👁️
- 1024 embedding dimensions 📊
- 4096 FFN hidden size 🔧
- Same architecture as TinyGPT2 (RoPE, GQA, RMSNorm, KV Cache, weight tying)
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~183M
- Training data: FineWeb-Edu (~8B tokens, streamed)
- Gradient checkpointing + 8-bit AdamW for memory efficiency

### TinyGPT2-SFT (Instruction Fine-Tuned) 💬

- **Base model**: TinyGPT2 (~95M parameters) or TinyGPT2.1 (~183M parameters)
- **Fine-tuning data**: Stanford Alpaca (52K) or SlimOrca (500K instruction-response pairs)
- **Training**: Response-only loss masking
- **Prompt format**: `### Instruction: ... ### Response: ...`
- **Capabilities**: Follows instructions, answers questions, writes creatively
- **Hardware**: Single NVIDIA RTX 3070 Ti (8GB VRAM)

### TinyGPT2-DPO (Preference Aligned) 🎯

- **Base model**: Any TinyGPT2-SFT checkpoint
- **Alignment data**: Anthropic HH-RLHF (~160K preference pairs)
- **Method**: Direct Preference Optimization (DPO)
- **Training**: Learns to prefer helpful/harmless responses over rejected ones
- **Hardware**: Single NVIDIA RTX 3070 Ti (8GB VRAM)

## Datasets 📖

| Model         | Dataset               | Tokens |
| ------------- | --------------------- | ------ |
| TinyGPT       | TinyStories           | ~300M  |
| TinyGPT-MoE   | TinyStories           | ~300M  |
| Wikipedia-MoE | Wikipedia (C4)        | ~500M  |
| TinyGPT2      | OpenWebText           | ~6.7B  |
| TinyGPT2.1    | FineWeb-Edu           | ~8B    |
| TinyGPT2-SFT  | Alpaca (52K) / SlimOrca (500K) | ~72M / ~500M |
| TinyGPT2-DPO  | Anthropic HH-RLHF    | ~160K pairs |

### Training Data Improvements 📈

- **Scale**: TinyGPT2 is trained on 6.7B+ tokens from OpenWebText; TinyGPT2.1 on ~8B tokens from FineWeb-Edu (educational web content).
- **Data Processing**: Efficient data loading with HuggingFace `datasets` and tiktoken tokenization for fast throughput.
- **Streaming**: FineWeb-Edu is streamed directly — no disk storage required.

## Installation 💿

To install TinyGPT, follow these steps:

```bash
# Clone the repository
git clone https://github.com/NotShrirang/tinygpt.git

# Navigate to the project directory
cd tinygpt

# Install the required packages
pip install -r requirements.txt

# Download the model weights
mkdir -p tinygpt/weights
```

### Liger Kernel Dependencies 🔧

For optimal training performance with **liger-kernel** (used by TinyGPT2 and MoE models), you need:

- **Linux operating system** (POSIX-compliant)
- **NVIDIA GPU with CUDA support**
- **liger-kernel**

```bash
# Install liger-kernel for training optimizations (Linux + CUDA only)
pip install liger-kernel
```

**Note**: On Windows or CPU-only environments, all models automatically fall back to pure PyTorch implementations without liger-kernel optimizations. The models will still work but training may be slower.

### Docker Support 🐳

TinyGPT fully supports Docker for easy deployment and development:

```bash
# Production deployment
docker-compose up --build

# Development with hot reload
docker-compose --profile dev up tinygpt-dev --build
```

The Docker setup includes:

- **Multi-model support**: All four model variants
- **Hot reload**: Automatic code updates during development
- **Cross-platform**: Works seamlessly on Windows, macOS, and Linux
- **Persistent storage**: Model weights are cached between container restarts

For detailed Docker usage, see `DOCKER.md`.

## Usage 🚀

### Model Selection 🎯

Choose from multiple model variants:

- **TinyGPT**: Standard 51M parameter model for story generation
- **TinyGPT-MoE**: 85M parameter MoE model with enhanced storytelling
- **Wikipedia-MoE**: 135M parameter MoE model trained on Wikipedia
- **TinyGPT2**: 95M parameter modern GPT with RoPE, GQA, and RMSNorm
- **TinyGPT2.1**: 183M parameter scaled model pretrained on FineWeb-Edu
- **TinyGPT2-SFT**: Instruction fine-tuned on Alpaca (52K) or SlimOrca (500K)
- **TinyGPT2-DPO**: Preference-aligned with Anthropic HH-RLHF

### Quick Start Options

#### Option 1: Load from HuggingFace 🤗

```python
from transformers import AutoModelForCausalLM
import tiktoken
import torch

# Load the instruction-tuned model
model = AutoModelForCausalLM.from_pretrained(
    "NotShrirang/tinygpt2-it",
    trust_remote_code=True,
)
model.eval()

# Tokenize and generate
enc = tiktoken.get_encoding("gpt2")
prompt = "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
input_ids = torch.tensor([enc.encode(prompt)])

with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=40)

print(enc.decode(output[0].tolist()))
```

#### Option 2: Streamlit Interface (Recommended for beginners)

```bash
streamlit run main.py
```

This launches a web application where you can:

- Select between all four model variants
- Adjust generation parameters (temperature, top-k, top-p, max tokens)
- Input text prompts and see real-time streaming responses
- Download models automatically from Hugging Face

#### Option 3: CLI Inference (TinyGPT2)

```bash
# Chat mode (default) — multi-turn conversation with memory
python inference.py --checkpoint checkpoints_sft/sft_epoch2.pth

# Instruction mode — single-turn, no memory
python inference.py --checkpoint checkpoints_sft/sft_epoch2.pth --mode instruction

# Raw mode — direct text completion, no template
python inference.py --checkpoint checkpoints/ckpt_step25500.pth --mode raw

# TinyGPT2.1 model
python inference.py --checkpoint checkpoints_v2.1/ckpt_step5000.pth --config v2.1

# Single prompt (non-interactive)
python inference.py --checkpoint checkpoints_sft/sft_epoch2.pth --prompt "What is the capital of France?"

# With custom settings
python inference.py --checkpoint checkpoints_sft/sft_epoch2.pth --max_tokens 200 --temperature 0.7 --top_k 40

# With a system prompt
python inference.py --checkpoint checkpoints_sft/sft_epoch2.pth --system "You are a helpful assistant."
```

Features:

- **Chat mode** (default): Multi-turn conversations with runtime memory — the model sees the full conversation history
- **Instruction mode**: Single-turn instruction following
- **Raw mode**: Direct text completion without any template
- KV cache for fast autoregressive generation
- Streaming token-by-token output with EOS detection
- Arrow key editing and up/down history navigation (persisted across sessions)
- Tab completion for slash commands
- Checkpoint info display (step, loss, tokens seen)

**Slash commands** (in interactive mode):

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/info` | Model info + current settings |
| `/mode <chat\|instruction\|raw>` | Switch mode |
| `/new` | Start a new chat (clears context) |
| `/temp <value>` | Set temperature |
| `/topk <value>` | Set top-k sampling |
| `/max <value>` | Set max generation tokens |
| `/system <text>` | Set/clear system prompt |
| `/history` | Show prompt history |
| `/clear` | Clear history and chat context |
| `/last` | Repeat the last generation |
| `/load <path>` | Load a different checkpoint |
| `/quit` | Exit |

#### Option 4: FastAPI Service (Production REST API)

```bash
# Start FastAPI server directly
python app.py

# Or use Docker
docker-compose up tinygpt-api --build
```

Features:

- REST API endpoints for text generation
- Multi-model support (TinyGPT, TinyGPT-MoE, TinyGPT2)
- Interactive Swagger docs at http://localhost:8000/docs
- Health monitoring and model management

For detailed API documentation, see `docs/API.md`.

#### Option 5: Docker (Recommended for production)

```bash
# Production deployment
docker-compose up --build

# Development mode with hot reload
docker-compose --profile dev up tinygpt-dev --build
```

Access the application at http://localhost:8501

### Cross-Platform Compatibility 🌐

TinyGPT runs smoothly on:

- **Windows** ✅ (with automatic fallback for liger-kernel)
- **macOS** ✅ (with automatic fallback for liger-kernel)
- **Linux** ✅ (full liger-kernel optimization support)
- **Docker** ✅ (all platforms)

## Training ⚙️

### TinyGPT / TinyGPT-MoE / Wikipedia-MoE

Trained using PyTorch on their respective datasets. See the training notebooks in the `notebooks/` directory.

<img src="https://github.com/user-attachments/assets/fd318849-d83b-4e44-aa3e-3119897cd4ae" alt="Loss Curve" width="70%"/>

### TinyGPT2 Pretraining

TinyGPT2 is pretrained on OpenWebText using `train_liger.py`:

```bash
# TinyGPT2 (95M) — default config
python train_liger.py

# TinyGPT2.1 (183M) — scaled config with FineWeb-Edu + 8-bit AdamW
python train_liger.py --config v2.1

# Resume from checkpoint
python train_liger.py --resume
python train_liger.py --config v2.1 --resume
```
<img width="1800" height="900" alt="training_loss_curve" src="https://github.com/user-attachments/assets/61eb397e-0473-4fcd-8d9e-4c695a523e94" />

Training configuration:

| | TinyGPT2 (95M) | TinyGPT2.1 (183M) |
|---|---|---|
| **Dataset** | OpenWebText | FineWeb-Edu (streamed) |
| **Tokens** | ~6.7B | ~8B |
| **Optimizer** | AdamW | 8-bit AdamW (bitsandbytes) |
| **Effective batch** | 262K tok/step | 262K tok/step |
| **Grad checkpointing** | No | Yes |
| **Mixed precision** | bfloat16 + torch.compile | bfloat16 + torch.compile |
| **Hardware** | RTX 3070 Ti (8GB) | RTX 3070 Ti (8GB) |

### Supervised Fine-Tuning (SFT)

Fine-tune TinyGPT2 on instruction-following tasks:

```bash
# Alpaca dataset (52K instructions) — default config
python sft.py --checkpoint checkpoints/ckpt_step25500.pth

# SlimOrca dataset (500K instructions) — v2.1 config
python sft.py --checkpoint checkpoints_v2.1/ckpt_step30000.pth --config v2.1 --dataset slimorca

# Resume SFT training
python sft.py --checkpoint checkpoints/ckpt_step25500.pth --resume
```

SFT configuration:

- **Hardware**: Single NVIDIA RTX 3070 Ti (8GB VRAM)
- **Datasets**: Stanford Alpaca (52K) or SlimOrca (500K instruction-response pairs)
- **Response-only loss masking**: Only trains on the response portion, not the instruction prompt
- **Prompt template**: `### Instruction: ... ### Input: ... ### Response: ...`

| Epoch | Train Loss | Val Loss | Train PPL | Val PPL |
| ----- | ---------- | -------- | --------- | ------- |
| 1     | 2.13       | 2.01     | 8.45      | 7.44    |
| 2     | 1.97       | 1.98     | 7.17      | 7.27    |
| 3     | 1.91       | 1.98     | 6.77      | 7.26    |

### DPO (Direct Preference Optimization)

Align the SFT model with human preferences using Anthropic's HH-RLHF dataset:

```bash
# Run DPO on an SFT checkpoint
python dpo.py --checkpoint checkpoints_sft/sft_epoch2.pth

# With custom hyperparameters
python dpo.py --checkpoint checkpoints_sft/sft_epoch2.pth --beta 0.05 --lr 5e-7 --epochs 1

# TinyGPT2.1 model
python dpo.py --checkpoint checkpoints_sft_v2.1/sft_epoch2.pth --config v2.1

# Resume DPO training
python dpo.py --checkpoint checkpoints_sft/sft_epoch2.pth --resume
```

DPO configuration:

- **Dataset**: Anthropic HH-RLHF (~160K preference pairs)
- **Method**: Trains a policy model against a frozen reference copy of the SFT model
- **Key metric**: Reward accuracy — how often the model prefers the chosen response over rejected
- **Emergency checkpointing**: Saves state on crash or Ctrl+C for safe resumability

### Training Optimizations 🚀

#### Standard TinyGPT Optimizations

- **Kernel Fusion**: Implemented to reduce memory bandwidth bottlenecks and speed up training operations
- **Mixed Precision Training**: Utilizes bfloat16 format for significantly faster training while maintaining numerical stability
- **Gradient Accumulation**: Applied to improve training stability and allow effective training with larger batch sizes
- **Cosine Scheduler**: Implements variable learning rate throughout training for better convergence
- **PyTorch's Multi-Head Attention**: Uses standard PyTorch implementations for Multi-Head Attention layers to boost training speed

#### TinyGPT2 Specific Optimizations

- **torch.compile**: Full model compilation for fused kernel execution
- **Grouped Query Attention (GQA)**: Query heads sharing KV groups — reduces memory while maintaining quality
- **Rotary Position Embeddings (RoPE)**: Efficient relative position encoding without learned position embeddings
- **RMSNorm**: Faster and more stable alternative to LayerNorm
- **KV Cache**: Efficient autoregressive generation — only computes attention for the new token
- **Weight Tying**: Shares weights between token embeddings and output projection to reduce parameters
- **Fused AdamW**: Uses CUDA-fused optimizer when available
- **8-bit AdamW** (v2.1): Halves optimizer state memory via bitsandbytes
- **Gradient Checkpointing** (v2.1): Trades compute for memory — enables larger models on limited VRAM
- **Emergency Checkpointing**: Saves model state on crash or Ctrl+C across all training scripts

#### MoE Specific Optimizations

- **liger-kernel Integration**: Uses optimized SwiGLU implementations for enhanced performance on Linux + CUDA
- **Expert Routing**: Dynamic routing of tokens to specialized experts for improved capabilities
- **Sparse Activation**: Only activates top-2 experts per token, maintaining efficiency while increasing model capacity
- **Automatic Fallback**: Gracefully falls back to PyTorch-native implementations on non-CUDA or Windows systems

## Project Structure 📁

```
tinygpt/
├── tinygpt/                  # Core package
│   ├── __init__.py           # Exports all models, configs, tokenizer
│   ├── model.py              # GPTLanguageModel, MoEGPTLanguageModel, WikipediaMoEGPTLanguageModel, TinyGPT2
│   ├── layers.py             # DecoderBlock, MoE blocks, GQA, RoPE, RMSNorm, TinyGPT2Block
│   ├── config.py             # GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig, TinyGPT2Config, TinyGPT2_1Config
│   ├── tokenizer.py          # Tiktoken-based tokenizer
│   ├── utils.py              # Generation utilities, mask helpers
│   └── weights/              # Model weight files
├── train_liger.py            # TinyGPT2/2.1 pretraining (OpenWebText / FineWeb-Edu)
├── sft.py                    # Supervised fine-tuning (Alpaca / SlimOrca)
├── dpo.py                    # DPO preference alignment (Anthropic HH-RLHF)
├── inference.py              # CLI inference — chat, instruction, and raw modes
├── main.py                   # Streamlit web UI (all models)
├── app.py                    # FastAPI REST API service
├── notebooks/                # Training notebooks
├── docs/                     # API documentation, Docker guide
├── docker-compose.yml        # Docker deployment
└── requirements.txt          # Python dependencies
```

## Deployment & System Requirements 💻

### Minimum Requirements (Inference)

- **CPU**: Any modern multi-core processor
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 1GB for model weights and dependencies
- **Python**: 3.8 or higher

### Optimal Performance (Training)

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM and CUDA 11.0+
- **RAM**: 16GB+
- **Additional**: liger-kernel for fused kernels

### Deployment Options

#### Local Development

```bash
# Standard Python environment
pip install -r requirements.txt
streamlit run main.py
```

#### Docker (Recommended)

```bash
# Production deployment
docker-compose up --build

# Development with auto-reload
docker-compose --profile dev up tinygpt-dev --build
```

#### Cloud Deployment

- **Streamlit Cloud**: Fully supported ✅
- **Heroku**: Supported with Docker ✅
- **AWS/GCP/Azure**: Supported with containerization ✅
- **Hugging Face Spaces**: Supported ✅

## Sample Outputs 📝

### TinyGPT (Standard Model)

```text
Prompt: One day, a dragon

Output:
One day, a dragon named Bobo was walking in the forest when he saw a little bunny. The bunny was sad because he had no friends. Bobo wanted to help the bunny, so he asked the bunny to give him a hug. The bunny said yes, and the bunny gave the bunny a hug.

Bobo was very happy and thanked the bunny. He named the bunny, and they became good friends. The bunny was always grateful for Bobo's help. They became good friends, and they always shared their toys and treats!
```

### TinyGPT2

```text
Prompt: The meaning of life

Output:
The meaning of life is more complex than its meanings. The two most common forms of human love are love and affection.

What is Love?

Love is both good and bad; it is one of love's most enduring possessions.

Love is the most fundamental, at times, measure of humanity's capacity for love. Love is an object of a man's desire and a desire. The desire of the man is the most important attribute of love.

Love is a self-awareness. It is a way of feeling out and doing something.
```

### TinyGPT2-SFT (Instruction Fine-Tuned)

```text
>>> Explain what machine learning is in simple terms.
Machine learning is a branch of computer science that focuses on using machine learning algorithms to identify patterns in data and identify patterns in data. It is a branch of computer science that focuses on creating computer systems that can perform tasks such as image recognition, image classification, and natural language processing. Machine learning algorithms are used to develop algorithms that can be used to generate and classify data in order to identify patterns in data. These algorithms are used to analyze large amounts of data and make predictions about future trends.

>>> What is the capital of France?
The capital of France is Paris.

>>> Write a motivational quote.
"The only way to make a difference is to be bold and courageous."
```

## License 📜

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

## Contributing 👥

Contributions are welcome! Feel free to submit pull requests, create issues, or suggest improvements to the model or codebase.

## Support ❤️

If you find TinyGPT useful, please consider starring the repository ⭐
