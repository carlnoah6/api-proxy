"""Configuration management for API Proxy"""
import json
import logging
import os
from datetime import timedelta, timezone
from pathlib import Path

# ── Timezone ──
SGT = timezone(timedelta(hours=8))

# ── Logging (must be before other imports that use log) ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api-proxy")

# ── Base directory (auto-detect: /app in Docker, repo root otherwise) ──
_BASE_DIR = os.environ.get("APP_BASE_DIR", str(Path(__file__).resolve().parent.parent))

# ── Core config ──
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8180"))
KEYS_FILE = Path(os.environ.get("KEYS_FILE", os.path.join(_BASE_DIR, "keys.json")))
MODELS_CONFIG_FILE = Path(os.environ.get("MODELS_CONFIG", os.path.join(_BASE_DIR, "models.json")))

# ── Lark OAuth config ──
LARK_APP_ID = os.environ.get("LARK_APP_ID", "cli_a90c3a6163785ed2")
LARK_APP_SECRET = os.environ.get("LARK_APP_SECRET")
if not LARK_APP_SECRET:
    log.warning("LARK_APP_SECRET is not set in environment variables. Lark API calls may fail.")
LARK_TOKEN_FILE = os.environ.get(
    "LARK_TOKEN_FILE",
    "/home/ubuntu/.openclaw/workspace/data/lark-user-token.json"
)

# ── Upstream Provider API Keys ──
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "sk-luna-2026-openclaw")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")

# ── API Key env mapping ──
_API_KEY_MAP = {
    "CLAUDE_API_KEY": CLAUDE_API_KEY,
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "KIMI_API_KEY": KIMI_API_KEY,
}


def load_models_config() -> list[dict]:
    """Load model registry from models.json.

    Returns a list of model dicts, each with a resolved 'api_key' field.
    """
    if not MODELS_CONFIG_FILE.exists():
        log.error(f"Models config not found: {MODELS_CONFIG_FILE}")
        return []

    try:
        data = json.loads(MODELS_CONFIG_FILE.read_text())
    except Exception as e:
        log.error(f"Failed to load models config: {e}")
        return []

    models = data.get("models", [])

    # Resolve API keys from environment
    for m in models:
        env_var = m.get("api_key_env", "")
        m["api_key"] = _API_KEY_MAP.get(env_var, os.environ.get(env_var, ""))

    return models


def get_models_registry() -> dict[str, dict]:
    """Return a dict mapping model_id -> model config (with resolved api_key)."""
    models = load_models_config()
    return {m["id"]: m for m in models}
