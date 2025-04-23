# TinyGPT 🤖

[![GitHub stars](https://img.shields.io/github/stars/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/NotShrirang/tinygpt?style=social)](https://github.com/NotShrirang/tinygpt/network/members)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tinygpt.streamlit.app/)

TinyGPT is a compact 50M parameter GPT model trained on a dataset of tiny stories, designed to generate coherent and creative text based on user input. ✨

## Features:


## Overview 🔍

TinyGPT is a lightweight GPT implementation trained on a dataset of short stories. With 50M parameters, it strikes a balance between computational efficiency and generative capability. The model was trained using a transformer architecture with self-attention mechanisms to capture contextual relationships in text.

## Model Architecture 🏗️

TinyGPT uses a standard GPT decoder-only transformer architecture with:

- 8 transformer blocks 🧱
- 8 attention heads 👁️
- 512 embedding dimensions 📊
- Vocabulary size of 50,304 tokens 📚
- Context window of 512 tokens 🪟

## Dataset 📖

The model was trained on the TinyStories dataset, a collection of short stories designed for training language models. This dataset provides simple narratives that help the model learn coherent story generation while maintaining a smaller size compared to larger language models.

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

## Usage 🚀

### Streamlit Interface 🖥️

The easiest way to interact with TinyGPT is through its Streamlit interface:

```bash
streamlit run main.py
```

This will launch a web application where you can input text and see the model's generated responses.

## Training ⚙️

TinyGPT was trained using PyTorch on the TinyStories dataset. The training process involved:

1. Tokenizing the input text
2. Creating sliding windows of fixed block size
3. Training the model with cross-entropy loss
4. Applying learning rate scheduling with warmup and cosine decay

<img src="https://github.com/user-attachments/assets/fd318849-d83b-4e44-aa3e-3119897cd4ae" alt="Loss Curve" width="70%"/>


For details on the training process, see the training notebook in the `notebooks/` directory.

## Sample Outputs 📝

### Example 1
```text
Prompt: One day, a dragon

Output:
One day, a dragon named Bobo was walking in the forest when he saw a little bunny. The bunny was sad because he had no friends. Bobo wanted to help the bunny, so he asked the bunny to give him a hug. The bunny said yes, and the bunny gave the bunny a hug.

Bobo was very happy and thanked the bunny. He named the bunny, and they became good friends. The bunny was always grateful for Bobo's help. They became good friends, and they always shared their toys and treats!
```

```
Prompt: A dog named

Output:
A dog named Max went for a walk. He saw a big tree and wanted to climb it. Max was very excited and started to climb the tree. He was very careful and did not fall.

Max saw a little girl named Sue. Sue was sad because she lost her toy. Max wanted to help Sue. He said, "Don't worry, Sue. I will help you find your toy."

Max and Sue looked for the toy together. They looked under the tree, behind the tree, and behind the tree. Finally, they found the toy under a big tree. Max was so happy and said, "Thank you, Sue! You are a good friend."

Sue and Max played with the toy all day. They were very happy and had a fun day!
```

## Inference 🔮

During inference, TinyGPT uses several techniques to produce high-quality text:

- Temperature scaling for controlling randomness
- Top-k and top-p sampling for focus and diversity
- Efficient token generation one at a time

## License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing 👥

Contributions are welcome! Feel free to submit pull requests, create issues, or suggest improvements to the model or codebase.

## Support ❤️

If you find TinyGPT useful, please consider starring the repository ⭐
