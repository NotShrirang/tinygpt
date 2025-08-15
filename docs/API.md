# TinyGPT FastAPI Service

A production-ready REST API for hosting TinyGPT and TinyGPT-MoE models with automatic model management, health monitoring, and comprehensive endpoints.

## Features

- 🚀 **Dual Model Support**: Host both TinyGPT and TinyGPT-MoE models
- 🔄 **Dynamic Model Loading**: Load/unload models on demand
- 📊 **Health Monitoring**: Real-time status and performance metrics
- 🛡️ **Error Handling**: Comprehensive error responses and validation
- 📚 **Auto Documentation**: Interactive Swagger/OpenAPI docs
- 🐳 **Docker Ready**: Full containerization support
- ⚡ **High Performance**: Optimized for production workloads

## Quick Start

### Option 1: Direct Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py

# Access the API at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Option 2: Docker (Recommended)

```bash
# Production deployment
docker-compose up tinygpt-api --build

# Development with hot reload
docker-compose --profile dev up tinygpt-api-dev --build

# API available at http://localhost:8000
```

## API Endpoints

### Core Endpoints

#### `GET /` - Root Information

Basic API information and navigation links.

#### `GET /health` - Health Check

```json
{
  "status": "healthy",
  "models_loaded": ["tinygpt", "tinygpt-moe"],
  "total_models": 2,
  "uptime": "2h 30m 45s"
}
```

#### `GET /models` - List Models

Returns detailed information about all available models:

```json
[
  {
    "name": "tinygpt",
    "description": "Standard 51M parameter GPT model for general story generation",
    "parameters": "51M",
    "loaded": true,
    "config": {
      "vocab_size": 50304,
      "block_size": 512,
      "n_embd": 512,
      "n_head": 8,
      "n_layer": 8
    }
  }
]
```

### Model Management

#### `POST /models/{model_name}/load` - Load Model

Loads a specific model into memory (background task).

#### `DELETE /models/{model_name}/unload` - Unload Model

Removes a model from memory to free resources.

### Text Generation

#### `POST /generate` - Generate Text

Main endpoint for text generation with comprehensive parameters:

**Request Body:**

```json
{
  "prompt": "Once upon a time",
  "model": "tinygpt",
  "max_new_tokens": 100,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.95,
  "word_repetition_penalty": 1.0
}
```

**Response:**

```json
{
  "generated_text": "Once upon a time, there was a brave little mouse...",
  "prompt": "Once upon a time",
  "model_used": "tinygpt",
  "tokens_generated": 45,
  "generation_time": 1.234,
  "tokens_per_second": 36.5,
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "word_repetition_penalty": 1.0
  }
}
```

## Usage Examples

### Python Client

```python
import requests

# Generate text
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "The dragon flew over",
    "model": "tinygpt-moe",
    "max_new_tokens": 80,
    "temperature": 0.8
})

result = response.json()
print(f"Generated: {result['generated_text']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A brave knight",
    "model": "tinygpt",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'

# Load a model
curl -X POST http://localhost:8000/models/tinygpt-moe/load
```

### JavaScript/Node.js

```javascript
const response = await fetch("http://localhost:8000/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    prompt: "In a magical forest",
    model: "tinygpt-moe",
    max_new_tokens: 100,
    temperature: 0.8,
  }),
});

const result = await response.json();
console.log(result.generated_text);
```

## Configuration

### Environment Variables

- `HOST`: API host (default: 0.0.0.0)
- `PORT`: API port (default: 8000)
- `PYTHONPATH`: Python path for imports

### Command Line Arguments

```bash
python app.py --help

Options:
  --host TEXT        Host to bind to (default: 0.0.0.0)
  --port INTEGER     Port to bind to (default: 8000)
  --reload          Enable auto-reload for development
  --workers INTEGER  Number of worker processes (default: 1)
```

## Docker Services

### Production Services

#### `tinygpt-api`

- **Port**: 8000
- **Features**: Production-optimized, health checks, persistent weights
- **Usage**: `docker-compose up tinygpt-api`

### Development Services

#### `tinygpt-api-dev`

- **Port**: 8001
- **Features**: Hot reload, full volume mount, development optimized
- **Usage**: `docker-compose --profile dev up tinygpt-api-dev`

## Performance

### Benchmarks

- **TinyGPT**: ~30-50 tokens/second (CPU)
- **TinyGPT-MoE**: ~25-40 tokens/second (CPU)
- **Memory Usage**: ~500MB-1GB per model
- **Cold Start**: ~10-30 seconds model loading

### Optimization Tips

1. **Pre-load Models**: Load models during startup for faster response times
2. **Memory Management**: Unload unused models to free memory
3. **Batch Requests**: Process multiple requests efficiently
4. **Caching**: Consider implementing response caching for common prompts

## Error Handling

### Common HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Model not found
- `503`: Service Unavailable (model not loaded)
- `500`: Internal Server Error

### Error Response Format

```json
{
  "detail": "Model 'invalid-model' not found"
}
```

## Monitoring

### Health Checks

- **Endpoint**: `/health`
- **Docker**: Built-in health checks with curl
- **Metrics**: Model status, uptime, memory usage

### Logging

The API provides structured logging for:

- Model loading/unloading events
- Generation requests and performance
- Error conditions and debugging

## Security Considerations

- **CORS**: Enabled for development (configure for production)
- **Rate Limiting**: Consider implementing for production use
- **Input Validation**: All inputs are validated with Pydantic
- **Model Access**: Models are isolated and safely managed

## Deployment

### Production Checklist

- [ ] Configure CORS policies
- [ ] Set up rate limiting
- [ ] Monitor resource usage
- [ ] Configure logging
- [ ] Set up health monitoring
- [ ] Secure model weights
- [ ] Configure reverse proxy (nginx/Apache)

### Scaling

- **Horizontal**: Deploy multiple API instances behind a load balancer
- **Vertical**: Increase memory/CPU for faster model loading
- **Model Distribution**: Use different instances for different models

## Troubleshooting

### Common Issues

1. **Models not loading**

   - Check model weights exist in `./tinygpt/weights/`
   - Verify file permissions
   - Check memory availability

2. **Slow generation**

   - Model not loaded (returns 503)
   - High temperature causing slower sampling
   - CPU-only inference (expected behavior)

3. **Memory issues**
   - Multiple models loaded simultaneously
   - Increase Docker memory limits
   - Unload unused models

### Debug Mode

```bash
# Enable debug logging
python app.py --reload

# Check Docker logs
docker-compose logs tinygpt-api
```

## Support

For issues and questions:

- 📖 Check the [main README](./README.md)
- 🐛 Open an issue on GitHub
- 📚 Review the interactive docs at `/docs`
- 🔍 Use the example client: `python api_client.py`
