import requests
import streamlit as st
import torch
import time
import os

from tinygpt.model import GPTLanguageModel, MoEGPTLanguageModel
from tinygpt.config import GPTConfig, MoEGPTConfig
from tinygpt.tokenizer import Tokenizer
from tinygpt.utils import generate


st.set_page_config(page_title="TinyGPT", page_icon=":robot_face:", layout="wide")

MODEL_CONFIGS = {
    "TinyGPT": {
        "class": GPTLanguageModel,
        "config": GPTConfig(),
        "local_path": "./tinygpt/weights/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt",
        "download_url": "https://huggingface.co/NotShrirang/tinygpt/resolve/main/final_model_tiny_stories_tiktoken_best22042025_1_weights.pt",
        "description": "A 50M parameter GPT model trained on a dataset of tiny stories."
    },
    "TinyGPT-MoE": {
        "class": MoEGPTLanguageModel,
        "config": MoEGPTConfig(),
        "local_path": "./tinygpt/weights/final_model_moe_storyteller_tiktoken_19072025.pt",
        "download_url": "https://huggingface.co/NotShrirang/tinygpt/resolve/main/final_model_moe_storyteller_tiktoken_19072025.pt",
        "description": "A Mixture of Experts GPT model with enhanced storytelling capabilities."
    }
}

def download_model(url, local_path):
    """Download model weights from URL to local path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB")
        
        progress_bar.empty()
        status_text.empty()

@st.cache_resource
def load_model(model_name):
    """Load the specified model."""
    model_config = MODEL_CONFIGS[model_name]
    model_class = model_config["class"]
    config = model_config["config"]
    local_path = model_config["local_path"]

    if not os.path.exists(local_path):
        st.error(f"Model weights not found at {local_path}")
        return None, None
    
    try:
        if model_name == "TinyGPT-MoE":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model_class.from_pretrained(local_path, device="cpu")
        else:
            model = model_class.from_pretrained(pretrained_model_path=local_path, device="cpu")
        
        tokenizer = Tokenizer()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def ensure_model_downloaded(model_name):
    """Ensure the model is downloaded, download if necessary."""
    model_config = MODEL_CONFIGS[model_name]
    local_path = model_config["local_path"]
    download_url = model_config["download_url"]
    
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading {model_name} model..."):
            st.info(f"Model weights not found. Downloading {model_name} from Hugging Face...")
            try:
                download_model(download_url, local_path)
                st.success(f"{model_name} downloaded successfully!")
                return True
            except Exception as e:
                st.error(f"Failed to download {model_name}: {e}")
                return False
    return True

def generate_text(model, input_tokens, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95, word_repetition_penalty=1.0):
    """Generate text using the model and input tokens."""
    for idx_next in generate(model, input_tokens, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, word_repetition_penalty=word_repetition_penalty):
        last_token = idx_next[:, -1]
        decoded_token = tokenizer.decode(last_token.tolist())
        if last_token.item() == tokenizer.eos_id:
            break
        yield decoded_token

with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox(
        "Choose Model:",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        help="Select which TinyGPT model to use for text generation"
    )
    
    # Display model description
    st.info(MODEL_CONFIGS[selected_model]["description"])
    
    # Model status
    model_path = MODEL_CONFIGS[selected_model]["local_path"]
    if os.path.exists(model_path):
        st.success(f"✅ {selected_model} is available")
    else:
        st.warning(f"⚠️ {selected_model} needs to be downloaded")
        if st.button(f"Download {selected_model}"):
            if ensure_model_downloaded(selected_model):
                st.rerun()
    
    st.divider()
    
    st.header("Generation Settings")
    st.text("Adjust the generation settings below:")
    temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.6, step=0.1)
    top_k = st.slider("Top K", min_value=0, max_value=100, value=50, step=1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    max_new_tokens = st.slider("Max New Tokens", min_value=20, max_value=500, value=100, step=20)
    
    if selected_model == "TinyGPT-MoE":
        st.info("💡 TinyGPT-MoE uses Mixture of Experts for enhanced storytelling capabilities.")
    
    st.info("Note: Adjusting these settings can affect the creativity and coherence of the generated text.")

st.title("TinyGPT :robot_face:")
st.markdown(f"Currently using: **{selected_model}**")
st.text(MODEL_CONFIGS[selected_model]["description"])

# Load the selected model
if not os.path.exists(MODEL_CONFIGS[selected_model]["local_path"]):
    st.warning(f"Please download {selected_model} from the sidebar first.")
    st.stop()

try:
    with st.spinner(f"Loading {selected_model}..."):
        model, tokenizer = load_model(selected_model)
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check the logs and try again.")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading {selected_model}: {e}")
    st.stop()

user_input = st.chat_input(placeholder="Type your message here...")

if user_input:
    prompt_tokens = tokenizer.encode(user_input, bos=False, eos=False)
    input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)
    
    with st.chat_message("user"):
        st.write(user_input)

    generated_text = generate_text(
        model, 
        input_tokens, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p
    )

    with st.chat_message("ai"):
        st.markdown(f"**{selected_model} is typing...**")
        start_time = time.time()
        text = st.write_stream(generated_text)
        end_time = time.time()
        
        tokens = tokenizer.encode(text, bos=False, eos=False)
        elapsed_time = end_time - start_time
        st.success(f"Finished generating **{len(tokens)}** tokens in **{elapsed_time:.2f} seconds**! **{len(tokens) / elapsed_time:.2f} tokens/s**")

with st.expander("Model Information", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("TinyGPT")
        st.write("- **Parameters**: 51M")
        st.write("- **Training Data**: Tiny Stories dataset")
        st.write("- **Architecture**: Standard GPT")
        st.write("- **Attention Heads**: 8")
        st.write("- **Embedding Dimension**: 512")

    with col2:
        st.subheader("TinyGPT-MoE")
        st.write("- **Parameters**: 85M")
        st.write("- **Training Data**: Tiny Stories dataset")
        st.write("- **Architecture**: Mixture of Experts GPT")
        st.write("- **Attention Heads**: 8")
        st.write("- **Experts**: 4")
        st.write("- **Embedding Dimension**: 512")
