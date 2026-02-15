"""Configuration management for API Proxy"""
import json
import logging
import os
from datetime import timedelta, timezone
from pathlib import Path

# ── Timezone ──
SGT = timezone(timedelta(hours=8))

# ── Logging ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api-proxy")

# ── Base directory ──
_BASE_DIR = os.environ.get("APP_BASE_DIR", str(Path(__file__).resolve().parent.parent))

# ── Core config ──
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8180"))
KEYS_FILE = Path(os.environ.get("KEYS_FILE", os.path.join(_BASE_DIR, "keys.json")))
MODELS_CONFIG_FILE = Path(os.environ.get("MODELS_CONFIG", os.path.join(_BASE_DIR, "models.json")))

# ── Lark OAuth config ──
LARK_APP_ID = os.environ.get("LARK_APP_ID", "cli_a90c3a6163785ed2")
LARK_APP_SECRET = os.environ.get("LARK_APP_SECRET")
if not LARK_APP_SECRET:
    log.warning("LARK_APP_SECRET is not set. Lark API calls may fail.")
LARK_TOKEN_FILE = os.environ.get(
    "LARK_TOKEN_FILE",
    "/home/ubuntu/.openclaw/workspace/data/lark-user-token.json",
)


def _load_models_json() -> dict:
    """Load the raw models.json file."""
    if not MODELS_CONFIG_FILE.exists():
        log.error(f"Models config not found: {MODELS_CONFIG_FILE}")
        return {"providers": {}, "known_models": []}
    try:
        return json.loads(MODELS_CONFIG_FILE.read_text())
    except Exception as e:
        log.error(f"Failed to load models config: {e}")
        return {"providers": {}, "known_models": []}


def get_providers() -> dict[str, dict]:
    """Return provider configs with resolved API keys."""
    data = _load_models_json()
    providers = data.get("providers", {})
    for pid, p in providers.items():
        env_var = p.get("api_key_env", "")
        p["api_key"] = os.environ.get(env_var, "")
        p["id"] = pid
    return providers


def get_known_models() -> list[dict]:
    """Return list of known model entries."""
    return _load_models_json().get("known_models", [])


def resolve_model(model_id: str, providers: dict) -> tuple[dict | None, str | None]:
    """Resolve a model ID to (provider_config, provider_id).

    1. Try exact match in known_models
    2. Try prefix match against provider model_prefixes
    Returns (None, None) if no match.
    """
    data = _load_models_json()

    # Exact match in known_models
    for m in data.get("known_models", []):
        if m["id"] == model_id:
            pid = m["provider"]
            if pid in providers:
                return providers[pid], pid
            return None, None

    # Prefix match against providers
    model_lower = model_id.lower()
    for pid, p in providers.items():
        for prefix in p.get("model_prefixes", []):
            if model_lower.startswith(prefix.lower()):
                return p, pid

    return None, None
