"""
FastAPI service for hosting TinyGPT and TinyGPT-MoE models.
Provides REST API endpoints for text generation with both model variants.
"""

import os
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import traceback
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from tinygpt import GPTLanguageModel, MoEGPTLanguageModel, TinyGPT2, GPTConfig, MoEGPTConfig, TinyGPT2Config, Tokenizer
from tinygpt.utils import generate


models: Dict[str, Any] = {}
tokenizer: Optional[Tokenizer] = None

MODEL_CONFIGS = {
    "tinygpt2": {
        "class": TinyGPT2,
        "config": TinyGPT2Config(),
        "local_path": "./tinygpt/weights/tinygpt2_ckpt_2026_02_21_8_15_it.pth",
        "description": "TinyGPT2 instruction fine-tuned on Stanford Alpaca (52K instructions)",
        "parameters": "95M",
    },
}


def load_model(model_name: str) -> bool:
    """Load a specific model into memory."""
    try:
        if model_name not in MODEL_CONFIGS:
            return False

        config = MODEL_CONFIGS[model_name]
        print(f"Loading {model_name}...")

        local_path = config["local_path"]
        if not os.path.exists(local_path):
            print(f"Model weights not found at {local_path}")
            return False

        model_class = config["class"]
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Loading {model_name} on GPU")
        else:
            device = torch.device("cpu")
            print(f"Loading {model_name} on CPU (this may be slow)")

        model = model_class.from_pretrained(local_path, device=device)
        model.eval()
        models[model_name] = model
        print(f"Successfully loaded {model_name}")
        return True

    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return False


def load_tokenizer() -> bool:
    """Load the tokenizer."""
    global tokenizer
    try:
        tokenizer = Tokenizer()
        print("Tokenizer loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI app."""
    print("Starting TinyGPT API service...")

    if not load_tokenizer():
        raise RuntimeError("Failed to load tokenizer")

    if not load_model("tinygpt2"):
        print("Warning: Failed to load default TinyGPT model")

    if not models:
        raise RuntimeError("No models could be loaded")
    
    print(f"API service started with {len(models)} model(s): {list(models.keys())}")
    yield

    print("Shutting down TinyGPT API service...")
    models.clear()

app = FastAPI(
    title="TinyGPT API",
    description="REST API for TinyGPT2 text generation models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation", min_length=1)
    model: str = Field(default="tinygpt2", description="Model to use: 'tinygpt2'")
    max_new_tokens: int = Field(default=100, ge=1, le=500, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.01, le=2.0, description="Sampling temperature (higher = more random)")
    top_k: int = Field(default=50, ge=0, le=100, description="Top-k sampling (0 = disabled)")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    word_repetition_penalty: float = Field(default=1.0, ge=0.1, le=2.0, description="Penalty for word repetition")


class TimingProfile(BaseModel):
    encode_ms: float = Field(..., description="Time to tokenize the prompt (ms)")
    time_to_first_token_ms: float = Field(..., description="Time from generation start to first token (ms)")
    generation_ms: float = Field(..., description="Time spent in the token generation loop (ms)")
    decode_ms: float = Field(..., description="Time to decode generated tokens back to text (ms)")
    total_ms: float = Field(..., description="End-to-end time for the request (ms)")
    avg_ms_per_token: float = Field(..., description="Average time per generated token (ms)")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")


class GenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    model_used: str
    tokens_generated: int
    generation_time: float
    tokens_per_second: float
    timings: TimingProfile
    parameters: Dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    description: str
    parameters: str
    loaded: bool
    config: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    total_models: int
    uptime: str

startup_time = time.time()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "TinyGPT API Service",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime_seconds = time.time() - startup_time
    uptime_str = f"{int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m {int(uptime_seconds % 60)}s"
    
    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=list(models.keys()),
        total_models=len(MODEL_CONFIGS),
        uptime=uptime_str
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models and their status."""
    model_list = []
    
    for name, config in MODEL_CONFIGS.items():
        model_info = ModelInfo(
            name=name,
            description=config["description"],
            parameters=config["parameters"],
            loaded=name in models,
            config={
                "vocab_size": config["config"].vocab_size,
                "block_size": config["config"].block_size,
                "n_embd": config["config"].n_embd,
                "n_head": config["config"].n_head,
                "n_layer": config["config"].n_layer,
            }
        )

        if hasattr(config["config"], "n_experts"):
            model_info.config.update({
                "n_experts": config["config"].n_experts,
                "top_experts": config["config"].top_experts
            })
        
        model_list.append(model_info)
    
    return model_list


@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str, background_tasks: BackgroundTasks):
    """Load a specific model into memory."""
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if model_name in models:
        return {"message": f"Model '{model_name}' is already loaded"}

    background_tasks.add_task(load_model, model_name)
    return {"message": f"Loading model '{model_name}' in background"}


@app.delete("/models/{model_name}/unload")
async def unload_model_endpoint(model_name: str):
    """Unload a specific model from memory."""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' is not loaded")
    
    del models[model_name]
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {"message": f"Model '{model_name}' unloaded successfully"}


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the specified model."""
    if request.model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {request.model}")
    
    if request.model not in models:
        raise HTTPException(status_code=503, detail=f"Model '{request.model}' is not loaded")
    
    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer is not available")
    
    try:
        request_start = time.perf_counter()

        encode_start = time.perf_counter()
        prompt_tokens = tokenizer.encode(request.prompt, bos=False, eos=False)
        input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu").unsqueeze(0)
        encode_time = time.perf_counter() - encode_start

        model = models[request.model]
        input_tokens = input_tokens.to(next(model.parameters()).device)

        generation_start = time.perf_counter()
        first_token_time = None
        generated_tokens = []
        eos_text = "<|endoftext|>"
        for idx_next in generate(
            model,
            input_tokens,
            request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            word_repetition_penalty=request.word_repetition_penalty
        ):
            if first_token_time is None:
                first_token_time = time.perf_counter() - generation_start
            last_token = idx_next[:, -1]
            if last_token.item() == tokenizer.eos_id:
                break
            generated_tokens.extend(last_token.tolist())
            # Check for literal <|endoftext|> from SFT models
            if len(generated_tokens) > 5:
                tail = tokenizer.decode(generated_tokens[-10:])
                if eos_text in tail:
                    break

        generation_time = time.perf_counter() - generation_start

        decode_start = time.perf_counter()
        generated_text = tokenizer.decode(generated_tokens)
        if eos_text in generated_text:
            generated_text = generated_text[:generated_text.index(eos_text)]
        decode_time = time.perf_counter() - decode_start

        total_time = time.perf_counter() - request_start
        tokens_generated = len(generated_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        avg_ms_per_token = (generation_time * 1000 / tokens_generated) if tokens_generated > 0 else 0
        ttft = first_token_time if first_token_time is not None else 0.0

        timings = TimingProfile(
            encode_ms=round(encode_time * 1000, 3),
            time_to_first_token_ms=round(ttft * 1000, 3),
            generation_ms=round(generation_time * 1000, 3),
            decode_ms=round(decode_time * 1000, 3),
            total_ms=round(total_time * 1000, 3),
            avg_ms_per_token=round(avg_ms_per_token, 3),
            prompt_tokens=len(prompt_tokens),
        )

        print(
            f"[profile] model={request.model} prompt_tokens={len(prompt_tokens)} "
            f"gen_tokens={tokens_generated} encode={timings.encode_ms}ms "
            f"ttft={timings.time_to_first_token_ms}ms gen={timings.generation_ms}ms "
            f"decode={timings.decode_ms}ms total={timings.total_ms}ms "
            f"tok/s={round(tokens_per_second, 2)}"
        )

        return GenerationResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            model_used=request.model,
            tokens_generated=tokens_generated,
            generation_time=round(generation_time, 3),
            tokens_per_second=round(tokens_per_second, 2),
            timings=timings,
            parameters={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "word_repetition_penalty": request.word_repetition_penalty
            }
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error during generation: {str(e)}\n{traceback_str}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/stream")
async def generate_text_stream(request: GenerationRequest):
    """Generate text with streaming response (for real-time generation)."""
    if request.model not in models:
        raise HTTPException(status_code=503, detail=f"Model '{request.model}' is not loaded")

    if not tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer is not available")

    async def event_stream():
        try:
            request_start = time.perf_counter()

            encode_start = time.perf_counter()
            prompt_tokens = tokenizer.encode(request.prompt, bos=False, eos=False)
            input_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
            encode_time = time.perf_counter() - encode_start

            model = models[request.model]
            generated_text = ""
            yielded_len = 0
            tokens_generated = 0
            eos_text = "<|endoftext|>"
            eos_buf_len = len(eos_text)

            generation_start = time.perf_counter()
            first_token_time = None
            stopped_on_eos_text = False
            for idx_next in generate(
                model,
                input_tokens,
                request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                word_repetition_penalty=request.word_repetition_penalty
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter() - generation_start
                last_token = idx_next[:, -1]
                if last_token.item() == tokenizer.eos_id:
                    break
                tokens_generated += 1
                decoded_token = tokenizer.decode(last_token.tolist())
                generated_text += decoded_token
                if eos_text in generated_text:
                    remaining = generated_text[yielded_len:generated_text.index(eos_text)]
                    if remaining:
                        yield remaining
                    stopped_on_eos_text = True
                    break
                safe = generated_text[:max(0, len(generated_text) - eos_buf_len)]
                if len(safe) > yielded_len:
                    yield safe[yielded_len:]
                    yielded_len = len(safe)

            if not stopped_on_eos_text and len(generated_text) > yielded_len:
                yield generated_text[yielded_len:]

            generation_time = time.perf_counter() - generation_start
            total_time = time.perf_counter() - request_start
            ttft = first_token_time if first_token_time is not None else 0.0
            print(
                f"[profile/stream] model={request.model} prompt_tokens={len(prompt_tokens)} "
                f"gen_tokens={tokens_generated} encode={round(encode_time * 1000, 3)}ms "
                f"ttft={round(ttft * 1000, 3)}ms gen={round(generation_time * 1000, 3)}ms "
                f"total={round(total_time * 1000, 3)}ms "
                f"tok/s={round(tokens_generated / generation_time, 2) if generation_time > 0 else 0}"
            )

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyGPT FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )
