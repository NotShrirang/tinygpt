"""
Example client for TinyGPT FastAPI service.
Demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any


class TinyGPTClient:
    """Client for TinyGPT API service."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model."""
        response = self.session.post(f"{self.base_url}/models/{model_name}/load")
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model."""
        response = self.session.delete(f"{self.base_url}/models/{model_name}/unload")
        response.raise_for_status()
        return response.json()
    
    def generate_text(
        self,
        prompt: str,
        model: str = "tinygpt",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        word_repetition_penalty: float = 1.0
    ) -> Dict[str, Any]:
        """Generate text using the specified model."""
        payload = {
            "prompt": prompt,
            "model": model,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "word_repetition_penalty": word_repetition_penalty
        }
        
        response = self.session.post(f"{self.base_url}/generate", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the TinyGPT API client."""
    
    # Initialize client
    client = TinyGPTClient()
    
    print("🤖 TinyGPT API Client Example\n")
    
    try:
        # Check health
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        print(f"   Uptime: {health['uptime']}\n")
        
        # List models
        print("2. Listing available models...")
        models = client.list_models()
        for model in models:
            print(f"   📦 {model['name']}: {model['description']}")
            print(f"      Parameters: {model['parameters']}, Loaded: {model['loaded']}")
        print()
        
        # Generate text with TinyGPT
        print("3. Generating text with TinyGPT...")
        prompt = "Once upon a time, there was a brave little mouse"
        
        result = client.generate_text(
            prompt=prompt,
            model="tinygpt",
            max_new_tokens=80,
            temperature=0.8
        )
        
        print(f"   Prompt: {result['prompt']}")
        print(f"   Generated: {result['generated_text']}")
        print(f"   Model: {result['model_used']}")
        print(f"   Tokens: {result['tokens_generated']}")
        print(f"   Speed: {result['tokens_per_second']} tokens/sec\n")
        
        # Try MoE model if available
        if any(model['name'] == 'tinygpt-moe' and model['loaded'] for model in models):
            print("4. Generating text with TinyGPT-MoE...")
            
            result_moe = client.generate_text(
                prompt=prompt,
                model="tinygpt-moe",
                max_new_tokens=80,
                temperature=0.8
            )
            
            print(f"   Prompt: {result_moe['prompt']}")
            print(f"   Generated: {result_moe['generated_text']}")
            print(f"   Model: {result_moe['model_used']}")
            print(f"   Tokens: {result_moe['tokens_generated']}")
            print(f"   Speed: {result_moe['tokens_per_second']} tokens/sec\n")
        else:
            print("4. TinyGPT-MoE model not available, skipping...\n")
        
        # Performance comparison
        print("5. Performance comparison...")
        test_prompt = "The dragon flew over the castle and"
        
        for model_name in ["tinygpt", "tinygpt-moe"]:
            if any(m['name'] == model_name and m['loaded'] for m in models):
                start_time = time.time()
                result = client.generate_text(
                    prompt=test_prompt,
                    model=model_name,
                    max_new_tokens=50,
                    temperature=0.7
                )
                end_time = time.time()
                
                print(f"   {model_name}:")
                print(f"   - Generation time: {result['generation_time']}s")
                print(f"   - Tokens/sec: {result['tokens_per_second']}")
                print(f"   - Total time: {end_time - start_time:.3f}s")
                print()
        
        print("✅ API client example completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to TinyGPT API")
        print("   Make sure the API server is running on http://localhost:8000")
        print("   Start with: python app.py or docker-compose up tinygpt-api")
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
