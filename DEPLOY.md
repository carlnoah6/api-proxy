# API Proxy Deployment Guide

## Architecture Overview

- **Proxy port**: 8180 (API Key authentication layer)
- **Upstream port**: 8080 (Upstream LLM Provider / OpenClaw)
- **Service management**: systemd (`api-proxy.service`)
- **Code location**: `/home/ubuntu/api-proxy/server.py`

---

## Pre-Change Backup

```bash
# Backup current code
cp /home/ubuntu/api-proxy/server.py /home/ubuntu/api-proxy/server.py.bak

# Backup configuration (keys, etc.)
cp /home/ubuntu/api-proxy/keys.json /home/ubuntu/api-proxy/keys.json.bak

# Timestamped backup (recommended)
cp /home/ubuntu/api-proxy/server.py "/home/ubuntu/api-proxy/server.py.bak.$(date +%Y%m%d_%H%M%S)"
```

---

## Testing

### Health Check (upstream connectivity)

```bash
curl http://localhost:8080/health
```

Expected: Returns JSON with `status: "ok"`, all accounts available.

### Model List (auth + proxy functionality)

```bash
curl http://localhost:8180/v1/models -H "x-api-key: <YOUR_API_KEY>"
```

Expected: Returns model list JSON.

### Admin Endpoints

```bash
# Fallback status
curl -H "x-api-key: sk-admin-luna2026" http://localhost:8180/admin/fallback

# Daily usage
curl -H "x-api-key: sk-admin-luna2026" http://localhost:8180/admin/usage/daily
```

### Full Functionality Test (send a request)

```bash
curl -X POST http://localhost:8180/v1/messages \
  -H "x-api-key: <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

---

## Deployment Steps

### Standard Restart (after code changes)

```bash
sudo systemctl restart api-proxy
```

### Verification

```bash
# 1. Check service status
systemctl status api-proxy

# 2. Health check
curl http://localhost:8080/health

# 3. Functional verification
curl http://localhost:8180/v1/models -H "x-api-key: <YOUR_API_KEY>"
```

### View Startup Logs

```bash
journalctl -u api-proxy --no-pager -n 30
```

---

## Rollback Plan

### Method 1: Restore Backup File

```bash
# Restore code
cp /home/ubuntu/api-proxy/server.py.bak /home/ubuntu/api-proxy/server.py

# Restart service
sudo systemctl restart api-proxy

# Verify
systemctl status api-proxy
curl http://localhost:8080/health
```

### Method 2: Git Rollback

```bash
cd /home/ubuntu/api-proxy
git log --oneline -5          # View recent commits
git checkout <commit> -- server.py  # Restore specific version
sudo systemctl restart api-proxy
```

---

## Log Viewing

```bash
# Follow logs in real time
journalctl -u api-proxy -f

# View last N lines
journalctl -u api-proxy --no-pager -n 50

# View today's logs
journalctl -u api-proxy --since today

# View specific time range
journalctl -u api-proxy --since "2026-02-11 00:00" --until "2026-02-11 23:59"

# View errors only
journalctl -u api-proxy -p err --no-pager
```

---

## Service Management Quick Reference

| Action | Command |
|--------|---------|
| Start | `sudo systemctl start api-proxy` |
| Stop | `sudo systemctl stop api-proxy` |
| Restart | `sudo systemctl restart api-proxy` |
| Status | `systemctl status api-proxy` |
| Enable on boot | `sudo systemctl enable api-proxy` |
| Disable on boot | `sudo systemctl disable api-proxy` |
| View logs | `journalctl -u api-proxy -f` |
| Reload config | `sudo systemctl daemon-reload` |

---

## Notes

- Service is configured with `Restart=on-failure`, auto-restarts after 5 seconds on crash
- Service is configured with `PYTHONUNBUFFERED=1`, logs are output in real time without buffering
- After modifying the service file, run `sudo systemctl daemon-reload`
- Legacy nohup startup method is deprecated; use systemctl for all service management
