# API Proxy (Luna API Proxy)

[![CI](https://github.com/carlnoah6/api-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/carlnoah6/api-proxy/actions/workflows/ci.yml)
[![Deploy](https://github.com/carlnoah6/api-proxy/actions/workflows/deploy.yml/badge.svg)](https://github.com/carlnoah6/api-proxy/actions/workflows/deploy.yml)

> **Status**: ✅ Running in production (Docker)
> **Port**: 8180

API Proxy is a FastAPI application that acts as a **multi-model gateway** between clients and upstream LLM providers. It handles authentication, usage tracking, and model-based routing.

## Architecture

```
Client → API Proxy (:8180) → Upstream LLM Providers
                              ├── Claude (Anthropic API)
                              ├── DeepSeek (OpenAI-compatible)
                              └── Kimi (OpenAI-compatible)
```

**Key Features:**
- **Multi-model Gateway**: Routes requests to the correct upstream based on the `model` field.
- **API Key Authentication**: Independent usage tracking per key.
- **Usage Statistics**: Tracks token usage by key, day, hour, and model.
- **No Format Conversion**: Each model uses its native API format (Anthropic or OpenAI).

## Model Registry

Models are configured in `models.json`:

| Model ID | Format | Upstream |
|----------|--------|----------|
| `claude-opus-4-6-thinking` | Anthropic Messages | Anthropic API |
| `deepseek-chat` | OpenAI Chat Completions | DeepSeek API |
| `kimi-k2.5` | OpenAI Chat Completions | Moonshot API |

### Routing

| Endpoint | Format | Models |
|----------|--------|--------|
| `POST /v1/messages` | Anthropic | `claude-opus-4-6-thinking` |
| `POST /v1/chat/completions` | OpenAI | `deepseek-chat`, `kimi-k2.5` |
| `GET /v1/models` | — | Lists all available models |

Clients must use the correct endpoint for the model's native format. The proxy does **not** convert between formats.

## Deployment (Docker)

### 1. Start / Update

```bash
docker compose pull && docker compose up -d
```

### 2. View Logs

```bash
docker compose logs -f
```

### 3. Configuration

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `KEYS_FILE` | `/app/keys.json` | Path to the keys file |
| `MODELS_CONFIG` | `/app/models.json` | Path to model registry config |
| `CLAUDE_API_KEY` | — | API key for Claude upstream |
| `DEEPSEEK_API_KEY` | — | API key for DeepSeek upstream |
| `KIMI_API_KEY` | — | API key for Kimi upstream |

**Volumes:**

- `./keys.json` → `/app/keys.json` (rw): API keys and usage data
- `./models.json` → `/app/models.json` (ro): Model registry

## Development

### Local Testing

```bash
pip install -r requirements.txt
pytest
```

### Code Structure (`src/`)

- `app.py`: FastAPI entrypoint, route definitions, proxy logic.
- `auth.py`: API key validation and management.
- `config.py`: Configuration loading and model registry.
- `usage.py`: Usage tracking and statistics recording.
- `health.py`: Health check endpoint.
- `admin.py`: Admin endpoints for key/usage management.

## API Documentation

### Admin Endpoints

```bash
export ADMIN_KEY="sk-admin-..."

# List models and status
curl -s -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/models

# View Daily Usage
curl -s -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage/daily

# List All Keys
curl -s -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys

# Create New Key
curl -X POST -H "x-api-key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "NewUser"}' http://localhost:8180/admin/keys
```

### Health Check

```bash
curl http://localhost:8180/health
```
