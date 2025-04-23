import streamlit as st

from tinygpt.model import GPTLanguageModel
from tinygpt.tokenizer import Tokenizer
from tinygpt.utils import generate
import torch
import requests


st.set_page_config(page_title="TinyGPT", page_icon=":robot_face:", layout="wide")


@st.cache_resource
def load_model():
    model = GPTLanguageModel.from_pretrained(pretrained_model_path="./tinygpt/weights/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt", device="cpu")
    tokenizer = Tokenizer()
    return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    with st.spinner("Downloading model..."):
        st.info("Model weights not found. Downloading from Hugging Face...")
        url = "https://huggingface.co/NotShrirang/tinygpt/resolve/main/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt"
        local_filename = "./tinygpt/weights/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success("Model downloaded successfully!")
        model, tokenizer = load_model()

def generate_text(model, input_tokens, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95, word_repetition_penalty=1.0):
    """Generate text using the model and input tokens."""
    for idx_next in generate(model, input_tokens, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, word_repetition_penalty=word_repetition_penalty):
        last_token = idx_next[:, -1]
        decoded_token = tokenizer.decode(last_token.tolist())
        if last_token.item() == tokenizer.eos_id:
            break

        yield decoded_token


st.title("TinyGPT :robot_face:")
st.text("A 50M parameter GPT model trained on a dataset of tiny stories.")

with st.sidebar:
    st.header("Settings")
    st.text("Adjust the generation settings below:")
    temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.6, step=0.1)
    top_k = st.slider("Top K", min_value=0, max_value=100, value=50, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    max_new_tokens = st.slider("Max New Tokens", min_value=20, max_value=500, value=100, step=20)
    st.info("Note: Adjusting these settings can affect the creativity and coherence of the generated text.")


user_input = st.chat_input(placeholder="Type your message here...")

if user_input:
    prompt_tokens = tokenizer.encode(user_input, bos=False, eos=False)

    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)
    with st.chat_message("user"):
        st.write(user_input)

    generated_text = generate_text(model, input_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)

    with st.chat_message("ai"):
        st.markdown("**TinyGPT is typing...**")
        st.write_stream(generated_text)
