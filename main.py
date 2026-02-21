import requests
import streamlit as st
import torch
import time
import os

from tinygpt import GPTLanguageModel, MoEGPTLanguageModel, WikipediaMoEGPTLanguageModel, TinyGPT2, GPTConfig, MoEGPTConfig, WikipediaMoEGPTConfig, TinyGPT2Config, Tokenizer
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
    },
    "Wikipedia-MoE": {
        "class": WikipediaMoEGPTLanguageModel,
        "config": WikipediaMoEGPTConfig(),
        "local_path": "./tinygpt/weights/final_model_moe_wikipedia_tiktoken_29072025.pt",
        "download_url": "https://huggingface.co/NotShrirang/tinygpt/resolve/main/fina_model_moe_wikipedia_180920245_ckpt_9553656.pt",  # No download URL yet as this is a new model
        "description": "A Wikipedia-trained MoE GPT model with 8 experts and 16 attention heads for enhanced knowledge representation."
    },
    "TinyGPT2": {
        "class": TinyGPT2,
        "config": TinyGPT2Config(),
        "local_path": "./tinygpt/weights/tinygpt2_ckpt_2026_02_18_20_42.pth",
        "download_url": "https://huggingface.co/NotShrirang/tinygpt/resolve/main/tinygpt2_ckpt_2026_02_18_20_42.pth",
        "description": "A 95M parameter GPT model with RoPE, GQA, and RMSNorm trained on OpenWebText.",
        "sft": False,
    },
    "TinyGPT2-SFT": {
        "class": TinyGPT2,
        "config": TinyGPT2Config(),
        "local_path": "./tinygpt/weights/tinygpt2_ckpt_2026_02_21_8_15_it.pth",
        "download_url": "https://huggingface.co/NotShrirang/tinygpt/resolve/main/tinygpt2_ckpt_2026_02_21_8_15_it.pth",
        "description": "TinyGPT2 instruction fine-tuned on Stanford Alpaca (52K instructions). Follows instructions and answers questions.",
        "sft": True,
    }
}


def format_sft_prompt(instruction, input_text=""):
    """Format a prompt for the instruction fine-tuned model using the Alpaca template."""
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name in ["TinyGPT-MoE", "Wikipedia-MoE", "TinyGPT2"]:
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
        if download_url is None:
            st.error(f"{model_name} model weights are not available for download yet. Please train the model first.")
            return False
            
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
    generated_text = ""
    yielded_len = 0
    eos_text = "<|endoftext|>"
    eos_buf_len = len(eos_text)
    for idx_next in generate(model, input_tokens, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, word_repetition_penalty=word_repetition_penalty):
        last_token = idx_next[:, -1]
        if last_token.item() == tokenizer.eos_id:
            break
        decoded_token = tokenizer.decode(last_token.tolist())
        generated_text += decoded_token
        # Stop on literal <|endoftext|> from SFT models
        if eos_text in generated_text:
            remaining = generated_text[yielded_len:generated_text.index(eos_text)]
            if remaining:
                yield remaining
            return
        # Hold back enough characters to detect <|endoftext|> spanning tokens
        safe = generated_text[:max(0, len(generated_text) - eos_buf_len)]
        if len(safe) > yielded_len:
            yield safe[yielded_len:]
            yielded_len = len(safe)
    # Flush remaining buffered text
    if len(generated_text) > yielded_len:
        yield generated_text[yielded_len:]

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
    elif selected_model == "Wikipedia-MoE":
        st.info("🧠 Wikipedia-MoE uses 8 experts and 16 attention heads for enhanced knowledge representation.")
    elif selected_model == "TinyGPT2-SFT":
        st.info("💬 TinyGPT2-SFT is instruction fine-tuned — just type your question or instruction naturally.")
    
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
    is_sft = MODEL_CONFIGS[selected_model].get("sft", False)
    if is_sft:
        full_prompt = format_sft_prompt(user_input)
    else:
        full_prompt = user_input
    prompt_tokens = tokenizer.encode(full_prompt, bos=False, eos=False)
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
        st.write("- **Parameters**: ~51M")
        st.write("- **Training Data**: Tiny Stories dataset")
        st.write("- **Architecture**: Standard GPT")
        st.write("- **Layers**: 8")
        st.write("- **Attention Heads**: 8")
        st.write("- **Embedding Dimension**: 512")

    with col2:
        st.subheader("TinyGPT-MoE")
        st.write("- **Parameters**: ~85M")
        st.write("- **Training Data**: Tiny Stories dataset")
        st.write("- **Architecture**: MoE with 4 experts (top-2)")
        st.write("- **Layers**: 8")
        st.write("- **Attention Heads**: 8")
        st.write("- **Embedding Dimension**: 512")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Wikipedia-MoE")
        st.write("- **Parameters**: ~135M")
        st.write("- **Training Data**: Wikipedia (C4 dataset)")
        st.write("- **Architecture**: MoE with 8 experts (top-2)")
        st.write("- **Layers**: 8")
        st.write("- **Attention Heads**: 16")
        st.write("- **Embedding Dimension**: 512")

    with col4:
        st.subheader("TinyGPT2")
        st.write("- **Parameters**: ~95M")
        st.write("- **Training Data**: OpenWebText")
        st.write("- **Architecture**: GPT with RoPE, GQA, RMSNorm")
        st.write("- **Layers**: 12")
        st.write("- **Attention Heads**: 12 (4 KV groups)")
        st.write("- **Embedding Dimension**: 768")
        st.write("- **FFN Hidden Size**: 2048")

    st.divider()

    st.subheader("TinyGPT2-SFT")
    st.write("- **Base Model**: TinyGPT2 (~95M parameters)")
    st.write("- **Fine-Tuning Data**: Stanford Alpaca (52K instruction-response pairs)")
    st.write("- **Training**: Instruction fine-tuned with response-only loss masking")
    st.write("- **Prompt Format**: `### Instruction: ... ### Response: ...`")
    st.write("- **Capabilities**: Follows instructions, answers questions, writes creatively")
