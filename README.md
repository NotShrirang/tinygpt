# TinyGPT 🤖

[![GitHub stars](https://img.shields.io/github/stars/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/network/members)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinygpt.streamlit.app/)

TinyGPT is a compact 50M parameter GPT model trained on a dataset of tiny stories, designed to generate coherent and creative text based on user input.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Streamlit Interface](#streamlit-interface)
  - [Python API](#python-api)
- [Training](#training)
- [Inference](#inference)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Overview

TinyGPT is a lightweight GPT implementation trained on a dataset of short stories. With 50M parameters, it strikes a balance between computational efficiency and generative capability. The model was trained using a transformer architecture with self-attention mechanisms to capture contextual relationships in text.

## Model Architecture

TinyGPT uses a standard GPT decoder-only transformer architecture with:

- 8 transformer blocks
- 8 attention heads
- 512 embedding dimensions
- Vocabulary size of 50,304 tokens
- Context window of 512 tokens

## Dataset

The model was trained on the TinyStories dataset, a collection of short stories designed for training language models. This dataset provides simple narratives that help the model learn coherent story generation while maintaining a smaller size compared to larger language models.

## Installation

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

## Usage

### Streamlit Interface

The easiest way to interact with TinyGPT is through its Streamlit interface:

```bash
streamlit run main.py
```

This will launch a web application where you can input text and see the model's generated responses.

### Python API

You can also use TinyGPT programmatically in your Python code:

```python
from tinygpt.model import GPTLanguageModel
from tinygpt.tokenizer import Tokenizer
from tinygpt.utils import generate
import torch

# Load the model and tokenizer
model = GPTLanguageModel.from_pretrained("./tinygpt/weights/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt", device="cpu")
tokenizer = Tokenizer()

# Prepare input
prompt = "One day, a little girl"
prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False)
input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)

# Generate text
for token in generate(model, input_tokens, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    # Process each generated token
    pass

# Or use the simpler wrapper function
from tinygpt.utils import generate_text
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)
```

## Training

TinyGPT was trained using PyTorch on the TinyStories dataset. The training process involved:

1. Tokenizing the input text
2. Creating sliding windows of fixed block size
3. Training the model with cross-entropy loss
4. Applying learning rate scheduling with warmup and cosine decay

For details on the training process, see the training notebook in the `notebooks/` directory.

## Inference

During inference, TinyGPT uses several techniques to produce high-quality text:

- Temperature scaling for controlling randomness
- Top-k and top-p sampling for focus and diversity
- Efficient token generation one at a time

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to submit pull requests, create issues, or suggest improvements to the model or codebase.

## Support

If you find TinyGPT useful, please consider starring the repository ⭐
