<img src="https://github.com/user-attachments/assets/8a90d976-57cb-4e3b-a9e7-33e37816eb81" alt="TinyGPT Banner" />

# TinyGPT 🤖

[![GitHub stars](https://img.shields.io/github/stars/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/network/members)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinygpt.streamlit.app/)

**TinyGPT** is an educational and production-ready implementation of the GPT (Generative Pre-trained Transformer) architecture, featuring two powerful model variants designed for creative text generation and storytelling. Built from the ground up with modern PyTorch, TinyGPT demonstrates how state-of-the-art language models can be both accessible and performant. ✨

🔗 **Quick Links:**

- 🤗 [HuggingFace Repository](https://huggingface.co/NotShrirang/tinygpt)
- 🚀 [Live Demo](https://tinygpt.streamlit.app/)
- 📚 [Training Notebooks](./notebooks/)

## Overview 🔍

TinyGPT represents a carefully crafted balance between **accessibility** and **performance** in language model design. This project showcases two distinct approaches to transformer architecture:

### 🎯 **Project Goals**

- **Educational**: Provide a clear, well-documented implementation of GPT architecture
- **Production-Ready**: Deliver a robust, efficient model suitable for real-world applications
- **Efficient**: Optimized for running on low-resource edge devices with minimal latency
- **Accessible**: Make it easy to run and deploy on various platforms

## Model Architecture 🏗️

TinyGPT comes in two variants:

### TinyGPT (Standard) 🤖

<img width="1748" height="1240" alt="TinyGPT Architecture" src="https://github.com/user-attachments/assets/1fbca234-61c9-4207-824a-763f97c903be" />

- 8 transformer blocks 🧱
- 8 attention heads 👁️
- 512 embedding dimensions 📊
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~51M

### TinyGPT-MoE (Mixture of Experts) 🧠

- 8 transformer blocks with MoE layers 🧱
- 8 attention heads 👁️
- 512 embedding dimensions 📊
- 4 experts per MoE layer with top-2 routing 🔀
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟
- Parameters: ~85M
- Enhanced storytelling capabilities through expert specialization

## Dataset 📖

The model was trained on the TinyStories dataset, a collection of short stories designed for training language models. This dataset provides simple narratives that help the model learn coherent story generation while maintaining a smaller size compared to larger language models.

### Training Data Improvements 📈

- **Scale**: TinyGPT was trained on approximately 300M tokens, significantly enhancing its language understanding capabilities.
- **Data Processing**: Initially faced challenges with data preprocessing pipelines that affected how data was passed to the model. These issues have been resolved, leading to more consistent and higher-quality training.

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

### MoE Model Dependencies 🔧

For the TinyGPT-MoE model to run with optimal performance (using **liger-kernel**), you need:

- **Linux operating system** (POSIX-compliant)
- **NVIDIA GPU with CUDA support**
- **liger-kernel==0.6.0**

```bash
# Install liger-kernel for MoE optimizations (Linux + CUDA only)
pip install liger-kernel==0.6.0
```

**Note**: On Windows or CPU-only environments, TinyGPT-MoE will automatically fall back to a PyTorch-native implementation without liger-kernel optimizations. The model will still work but may be slower.

### Docker Support 🐳

TinyGPT now fully supports Docker for easy deployment and development:

```bash
# Production deployment
docker-compose up --build

# Development with hot reload
docker-compose --profile dev up tinygpt-dev --build
```

The Docker setup includes:

- **Multi-model support**: Both TinyGPT and TinyGPT-MoE
- **Hot reload**: Automatic code updates during development
- **Cross-platform**: Works seamlessly on Windows, macOS, and Linux
- **Persistent storage**: Model weights are cached between container restarts

For detailed Docker usage, see `DOCKER.md`.

## Usage 🚀

### Model Selection 🎯

Choose between two model variants:

- **TinyGPT**: Standard 51M parameter model for general story generation
- **TinyGPT-MoE**: 85M parameter Mixture of Experts model with enhanced storytelling capabilities

### Quick Start Options

#### Option 1: Streamlit Interface (Recommended for beginners)

```bash
streamlit run main.py
```

This launches a web application where you can:

- Select between TinyGPT and TinyGPT-MoE models
- Adjust generation parameters (temperature, top-k, top-p)
- Input text prompts and see real-time generated responses
- Download models automatically from Hugging Face

#### Option 2: FastAPI Service (Production REST API)

```bash
# Start FastAPI server directly
python app.py

# Or use Docker
docker-compose up tinygpt-api --build
```

Features:

- REST API endpoints for text generation
- Dual model support (TinyGPT and TinyGPT-MoE)
- Interactive Swagger docs at http://localhost:8000/docs
- Health monitoring and model management

For detailed API documentation, see `docs/API.md`.

#### Option 3: Docker (Recommended for production)

```bash
# Production deployment
docker-compose up --build

# Development mode with hot reload
docker-compose --profile dev up tinygpt-dev --build
```

Access the application at http://localhost:8501

### Cross-Platform Compatibility 🌐

TinyGPT runs smoothly on:

- **Windows** ✅ (with automatic fallback for MoE models)
- **macOS** ✅ (with automatic fallback for MoE models)
- **Linux** ✅ (full liger-kernel optimization support)
- **Docker** ✅ (all platforms)

## Deployment & System Requirements 💻

### Minimum Requirements

- **CPU**: Any modern multi-core processor
- **RAM**: 4GB+ (8GB recommended)
- **Storage**: 1GB for model weights and dependencies
- **Python**: 3.8 or higher

### Optimal Performance (TinyGPT-MoE)

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.0+
- **RAM**: 8GB+
- **Additional**: liger-kernel==0.6.0

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

## Training ⚙️

TinyGPT was trained using PyTorch on the TinyStories dataset. The training process involved:

1. Tokenizing the input text
2. Creating sliding windows of fixed block size
3. Training the model with cross-entropy loss
4. Applying learning rate scheduling with warmup and cosine decay

<img src="https://github.com/user-attachments/assets/fd318849-d83b-4e44-aa3e-3119897cd4ae" alt="Loss Curve" width="70%"/>

### Training Optimizations 🚀

TinyGPT's training process leverages several optimization techniques to enhance speed, stability, and performance:

#### Standard TinyGPT Optimizations

- **Kernel Fusion**: Implemented to reduce memory bandwidth bottlenecks and speed up training operations
- **Mixed Precision Training**: Utilizes bfloat16 format for significantly faster training while maintaining numerical stability
- **Gradient Accumulation**: Applied to improve training stability and allow effective training with larger batch sizes
- **Cosine Scheduler**: Implements variable learning rate throughout training for better convergence
- **PyTorch's Multi-Head Attention**: Uses standard PyTorch implementations for Multi-Head Attention layers to boost training speed

#### TinyGPT-MoE Specific Optimizations

- **liger-kernel Integration**: Uses optimized SwiGLU implementations for enhanced performance on Linux + CUDA
- **Expert Routing**: Dynamic routing of tokens to specialized experts for improved storytelling capabilities
- **Sparse Activation**: Only activates top-2 experts per token, maintaining efficiency while increasing model capacity
- **Automatic Fallback**: Gracefully falls back to PyTorch-native implementations on non-CUDA or Windows systems

While using PyTorch's native attention implementation deviates from the "from scratch" philosophy, it enables more rapid model iteration and training with available resources.

For details on the training process, see the training notebook in the `notebooks/` directory.

## Sample Outputs 📝

### TinyGPT (Standard Model)

#### Example 1

```text
Prompt: One day, a dragon

Output:
One day, a dragon named Bobo was walking in the forest when he saw a little bunny. The bunny was sad because he had no friends. Bobo wanted to help the bunny, so he asked the bunny to give him a hug. The bunny said yes, and the bunny gave the bunny a hug.

Bobo was very happy and thanked the bunny. He named the bunny, and they became good friends. The bunny was always grateful for Bobo's help. They became good friends, and they always shared their toys and treats!
```

#### Example 2

```text
Prompt: A dog named

Output:
A dog named Max went for a walk. He saw a big tree and wanted to climb it. Max was very excited and started to climb the tree. He was very careful and did not fall.

Max saw a little girl named Sue. Sue was sad because she lost her toy. Max wanted to help Sue. He said, "Don't worry, Sue. I will help you find your toy."

Max and Sue looked for the toy together. They looked under the tree, behind the tree, and behind the tree. Finally, they found the toy under a big tree. Max was so happy and said, "Thank you, Sue! You are a good friend."

Sue and Max played with the toy all day. They were very happy and had a fun day!
```

## License 📜

This project is licensed under the GPL-3.0 license - see the LICENSE file for details.

## Contributing 👥

Contributions are welcome! Feel free to submit pull requests, create issues, or suggest improvements to the model or codebase.

## Support ❤️

If you find TinyGPT useful, please consider starring the repository ⭐
