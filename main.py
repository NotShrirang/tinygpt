import streamlit as st

from tinygpt.model import GPTLanguageModel
from tinygpt.tokenizer import Tokenizer
from tinygpt.utils import generate
import torch


st.set_page_config(page_title="TinyGPT", page_icon=":robot_face:", layout="wide")


@st.cache_resource
def load_model():
    model = GPTLanguageModel.from_pretrained(pretrained_model_path="./tinygpt/weights/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt", device="cpu")
    tokenizer = Tokenizer()
    return model, tokenizer

model, tokenizer = load_model()


def generate_text(model, input_tokens, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
    """Generate text using the model and input tokens."""
    for idx_next in generate(model, input_tokens, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p):
        last_token = idx_next[:, -1]
        decoded_token = tokenizer.decode(last_token.tolist())
        if last_token.item() == tokenizer.eos_id:
            break

        yield decoded_token


st.title("TinyGPT :robot_face:")
st.text("A 50M parameter GPT model trained on a dataset of tiny stories.")

user_input = st.chat_input(placeholder="Type your message here...")

if user_input:
    prompt_tokens = tokenizer.encode(user_input, bos=False, eos=False)

    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)
    with st.chat_message("user"):
        st.write(user_input)

    generated_text = generate_text(model, input_tokens, 100, temperature=0.8, top_k=50, top_p=0.95)

    with st.chat_message("ai"):
        st.markdown("**TinyGPT is typing...**")
        st.write_stream(generated_text)
