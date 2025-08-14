# Docker Usage Guide for TinyGPT

## Quick Start

### Production Mode

```bash
# Build and run the application
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The application will be available at `http://localhost:8501`

### Development Mode

```bash
# Run with hot reload enabled
docker-compose --profile dev up tinygpt-dev --build
```

The development server will be available at `http://localhost:8501` with file watching enabled.

## Available Services

### tinygpt-app (Production)

- **Port**: 8501
- **Features**:
  - Optimized for production use
  - Persistent model weights storage
  - Health checks enabled
  - Auto-restart on failure

### tinygpt-dev (Development)

- **Port**: 8501
- **Features**:
  - Hot reload on file changes
  - Full project volume mount
  - Poll-based file watching for cross-platform compatibility

## Docker Commands

### Basic Operations

```bash
# Start services
docker-compose up

# Start services in background
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build

# View logs
docker-compose logs

# View logs for specific service
docker-compose logs tinygpt-app
```

### Development Operations

```bash
# Start development environment
docker-compose --profile dev up

# Start only development service
docker-compose up tinygpt-dev

# Enable development override with hot reload
docker-compose -f docker-compose.yml -f docker-compose.override.yml up
```

### Maintenance

```bash
# Remove containers and networks
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove containers, networks, volumes, and images
docker-compose down -v --rmi all

# Clean up everything Docker-related
docker system prune -a
```

## Volume Mounts

### Production Mode

- `./tinygpt/weights:/app/tinygpt/weights` - Persists downloaded model weights

### Development Mode

- `.:/app` - Full project mount for live code editing

## Environment Variables

- `STREAMLIT_SERVER_PORT=8501` - Server port
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0` - Bind to all interfaces
- `STREAMLIT_SERVER_HEADLESS=true` - Run without opening browser
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` - Disable telemetry
- `STREAMLIT_SERVER_RUN_ON_SAVE=true` - Enable hot reload (dev mode)

## Health Checks

The production service includes health checks that:

- Check application availability every 30 seconds
- Allow 40 seconds for initial startup
- Retry 3 times before marking as unhealthy
- Use curl to test the Streamlit health endpoint

## Troubleshooting

### Common Issues

1. **Port already in use**

   ```bash
   # Check what's using the port
   netstat -ano | findstr :8501
   # Kill the process or change the port in docker-compose.yml
   ```

2. **Model not downloading**

   - Ensure internet connectivity
   - Check the weights directory permissions
   - Clear Docker volumes and restart

3. **Hot reload not working**

   - Use the development profile: `docker-compose --profile dev up`
   - Or use the override file: `docker-compose -f docker-compose.yml -f docker-compose.override.yml up`

4. **Container won't start**

   ```bash
   # Check logs
   docker-compose logs tinygpt-app

   # Rebuild from scratch
   docker-compose down -v
   docker-compose build --no-cache
   docker-compose up
   ```

### Performance Tips

1. **First run**: The first startup may be slow due to model download
2. **Subsequent runs**: Model weights are cached in the volume
3. **Memory**: Ensure Docker has enough memory allocated (4GB+ recommended)
4. **Storage**: The model weights file is ~200MB

## Network Configuration

- **Network name**: `tinygpt-network`
- **External access**: Services are accessible via the host ports
- **Internal communication**: Services can communicate using service names
