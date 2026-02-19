<img src="https://github.com/user-attachments/assets/8a90d976-57cb-4e3b-a9e7-33e37816eb81" alt="TinyGPT Banner" />

# TinyGPT 🤖

[![GitHub stars](https://img.shields.io/github/stars/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/network/members)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinygpt.streamlit.app/)

**TinyGPT** is an educational and production-ready implementation of the GPT (Generative Pre-trained Transformer) architecture, featuring four model variants ranging from simple story generators to a modern GPT-2 class model with RoPE, GQA, and RMSNorm. Built from the ground up with modern PyTorch, TinyGPT demonstrates how state-of-the-art language models can be both accessible and performant. ✨

🔗 **Quick Links:**

- 🤗 [HuggingFace Repository](https://huggingface.co/NotShrirang/tinygpt)
- 🚀 [Live Demo](https://tinygpt.streamlit.app/)
- 📚 [Training Notebooks](./notebooks/)

## Overview 🔍

TinyGPT represents a carefully crafted balance between **accessibility** and **performance** in language model design. The project progresses through four model variants — from a standard GPT to Mixture-of-Experts architectures to a modern GPT-2 model with cutting-edge techniques.

### 🎯 **Project Goals**

- **Educational**: Provide a clear, well-documented implementation of GPT architecture
- **Production-Ready**: Deliver robust, efficient models suitable for real-world applications
- **Efficient**: Optimized for running on consumer GPUs and edge devices with minimal latency
- **Accessible**: Make it easy to run, train, fine-tune, and deploy on various platforms

## Model Architecture 🏗️

TinyGPT comes in four variants:

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
- Training data: OpenWebText (~3.4B+ tokens)

## Datasets 📖

| Model         | Dataset        | Tokens |
| ------------- | -------------- | ------ |
| TinyGPT       | TinyStories    | ~300M  |
| TinyGPT-MoE   | TinyStories    | ~300M  |
| Wikipedia-MoE | Wikipedia (C4) | ~500M  |
| TinyGPT2      | OpenWebText    | ~3.4B+ |

### Training Data Improvements 📈

- **Scale**: TinyGPT2 is trained on 3.4B+ tokens from OpenWebText, significantly enhancing its general language understanding.
- **Data Processing**: Efficient data loading with HuggingFace `datasets` and tiktoken tokenization for fast throughput.

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

Choose from four model variants:

- **TinyGPT**: Standard 51M parameter model for story generation
- **TinyGPT-MoE**: 85M parameter MoE model with enhanced storytelling
- **Wikipedia-MoE**: 135M parameter MoE model trained on Wikipedia
- **TinyGPT2**: 95M parameter modern GPT with RoPE, GQA, and RMSNorm

### Quick Start Options

#### Option 1: Streamlit Interface (Recommended for beginners)

```bash
streamlit run main.py
```

This launches a web application where you can:

- Select between all four model variants
- Adjust generation parameters (temperature, top-k, top-p, max tokens)
- Input text prompts and see real-time streaming responses
- Download models automatically from Hugging Face

#### Option 2: CLI Inference (TinyGPT2)

```bash
# Single prompt
python inference.py --checkpoint checkpoints/ckpt_step13000.pth --prompt "The meaning of life"

# Interactive mode
python inference.py --checkpoint checkpoints/ckpt_step13000.pth

# With custom settings
python inference.py --checkpoint checkpoints/ckpt_step13000.pth --max_tokens 200 --temperature 0.8 --top_k 40
```

Features:

- KV cache for fast autoregressive generation
- Streaming token-by-token output
- Interactive REPL mode
- Checkpoint info display (step, loss, tokens seen)

#### Option 3: FastAPI Service (Production REST API)

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

#### Option 4: Docker (Recommended for production)

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
# Start training from scratch
python train_liger.py

# Resume from checkpoint
python train_liger.py --resume
```

Training configuration:

- **Hardware**: Single NVIDIA RTX 3070 Ti (8GB VRAM)
- **Effective batch size**: 262K tokens/step (batch 8 × grad accum 64 × block size 512)
- **Optimizer**: AdamW with cosine decay schedule and warmup
- **Mixed precision**: bfloat16 with `torch.compile` for speed
- **Evaluation**: Periodic validation with sample text generation
- **Checkpointing**: Automatic saves with train/val loss tracking

### Training Optimizations 🚀

#### Standard TinyGPT Optimizations

- **Kernel Fusion**: Implemented to reduce memory bandwidth bottlenecks and speed up training operations
- **Mixed Precision Training**: Utilizes bfloat16 format for significantly faster training while maintaining numerical stability
- **Gradient Accumulation**: Applied to improve training stability and allow effective training with larger batch sizes
- **Cosine Scheduler**: Implements variable learning rate throughout training for better convergence
- **PyTorch's Multi-Head Attention**: Uses standard PyTorch implementations for Multi-Head Attention layers to boost training speed

#### TinyGPT2 Specific Optimizations

- **torch.compile**: Full model compilation for fused kernel execution
- **Grouped Query Attention (GQA)**: 12 query heads sharing 4 KV groups — reduces memory while maintaining quality
- **Rotary Position Embeddings (RoPE)**: Efficient relative position encoding without learned position embeddings
- **RMSNorm**: Faster and more stable alternative to LayerNorm
- **KV Cache**: Efficient autoregressive generation — only computes attention for the new token
- **Weight Tying**: Shares weights between token embeddings and output projection to reduce parameters
- **Fused AdamW**: Uses CUDA-fused optimizer when available

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
│   ├── config.py             # GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig, TinyGPT2Config
│   ├── tokenizer.py          # Tiktoken-based tokenizer
│   ├── utils.py              # Generation utilities, mask helpers
│   └── weights/              # Model weight files
├── train_liger.py            # TinyGPT2 pretraining script (OpenWebText)
├── inference.py              # TinyGPT2 CLI inference with KV cache
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

## License 📜

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

## Contributing 👥

Contributions are welcome! Feel free to submit pull requests, create issues, or suggest improvements to the model or codebase.

## Support ❤️

If you find TinyGPT useful, please consider starring the repository ⭐
