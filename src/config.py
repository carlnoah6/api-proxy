"""Configuration management for API Proxy"""
import logging
import os
from datetime import timedelta, timezone
from pathlib import Path

# ── Timezone ──
SGT = timezone(timedelta(hours=8))

# ── Base directory (auto-detect: /app in Docker, repo root otherwise) ──
_BASE_DIR = os.environ.get("APP_BASE_DIR", str(Path(__file__).resolve().parent.parent))

# ── Core config ──
UPSTREAM = os.environ.get("UPSTREAM_URL", "http://localhost:8080")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8180"))
KEYS_FILE = Path(os.environ.get("KEYS_FILE", os.path.join(_BASE_DIR, "keys.json")))
FALLBACK_CONFIG_FILE = Path(os.environ.get("FALLBACK_CONFIG", os.path.join(_BASE_DIR, "fallback.json")))

# ── Lark OAuth config ──
LARK_APP_ID = "cli_a90c3a6163785ed2"
LARK_APP_SECRET = "IpWX3GqEgSDYfDVD8ICUedxjfbIanr7O"
LARK_TOKEN_FILE = os.environ.get(
    "LARK_TOKEN_FILE",
    "/home/ubuntu/.openclaw/workspace/data/lark-user-token.json"
)

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api-proxy")
