# API Proxy (Luna API Proxy)

[![CI](https://github.com/carlnoah6/api-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/carlnoah6/api-proxy/actions/workflows/ci.yml)
[![Deploy](https://github.com/carlnoah6/api-proxy/actions/workflows/deploy.yml/badge.svg)](https://github.com/carlnoah6/api-proxy/actions/workflows/deploy.yml)

> Code location: `/home/ubuntu/api-proxy/server.py`
> Ports: **8180** (proxy layer) → **8080** (upstream LLM provider)
> Status: ✅ Running in production

## Architecture

```
OpenClaw → api-proxy (8180) → Upstream LLM Provider (8080) → Claude/Gemini API
```

API Proxy is a FastAPI application that sits in front of the upstream LLM provider, providing:
- **API Key Authentication**: One key per user with independent usage tracking
- **Smart Fallback**: Proactively monitors model quotas and auto-switches when limits are reached
- **Usage Statistics**: Multi-dimensional tracking by key/day/hour/model
- **Format Conversion**: Anthropic ↔ OpenAI format conversion (supports streaming)
- **OAuth Callbacks**: Handles Lark calendar authorization + card action callbacks

## Common Commands

### Start / Restart

```bash
# Check running processes
ps aux | grep api-proxy

# Restart
kill $(pgrep -f "api-proxy/server.py")
nohup python3 /home/ubuntu/api-proxy/server.py > /home/ubuntu/api-proxy/server.log 2>&1 &
```

### Health Check

```bash
# Proxy layer health
curl http://localhost:8180/health

# Upstream LLM provider health (includes per-model quota details)
curl http://localhost:8080/health
```

### Admin Endpoints (requires Admin Key)

```bash
ADMIN_KEY="sk-admin-luna2026"

# View fallback status (per-model quotas + availability)
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/fallback

# List all API keys
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys

# View total usage
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage

# Daily usage stats (with model breakdown)
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage/daily

# Hourly usage stats
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage/hourly

# Detailed usage for a specific key
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/usage

# Create a new key
curl -X POST -H "x-api-key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "NewUser"}' http://localhost:8180/admin/keys

# Disable / Enable a key
curl -X POST -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/disable
curl -X POST -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/enable
```

## Configuration Files

| File | Description |
|------|-------------|
| `server.py` | Main service code |
| `keys.json` | API key data (includes usage stats) |
| `fallback.json` | Fallback chain configuration |
| `server.log` | Runtime log |

## Fallback Logic

1. Background thread polls upstream `/health` every 30s, caching per-model quotas
2. On incoming request, checks target model quota; if < 5%, auto-switches to next available tier
3. Quota aggregation: iterates all upstream accounts, takes the max remaining quota per model
4. Reactive fallback: even if proactive check passes, if upstream returns 429/503/exhausted, attempts to switch

## Known Issues & Fix Log

### 2026-02-11: Multi-account Quota Aggregation Bug
- **Issue**: `HealthCache.poll()` only read `accounts[0]`, ignoring subsequent accounts
- **Fix**: Changed to iterate all accounts, taking the max `remainingFraction` per model
- **Impact**: Newly added account (carlnoah6) quota was not recognized, causing Claude to be incorrectly judged as 0% and triggering fallback

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `http://localhost:8080` | Upstream LLM provider address |
| `PROXY_PORT` | `8180` | Proxy listening port |
| `KEYS_FILE` | `/home/ubuntu/api-proxy/keys.json` | Key data file |
| `FALLBACK_CONFIG` | `/home/ubuntu/api-proxy/fallback.json` | Fallback configuration file |
