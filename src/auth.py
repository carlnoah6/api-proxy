"""API Key authentication and management"""
import json
import uuid
from datetime import datetime

from fastapi import HTTPException, Request

from .config import KEYS_FILE, SGT


def load_keys() -> dict:
    """Load keys data from file"""
    if KEYS_FILE.exists():
        return json.loads(KEYS_FILE.read_text())
    return {"admin_key": "sk-admin-luna2026", "keys": {}}


def save_keys(data: dict):
    """Save keys data to file"""
    KEYS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def get_key_info(api_key: str) -> dict | None:
    """Get info for a specific API key"""
    data = load_keys()
    if api_key in data["keys"]:
        return data["keys"][api_key]
    return None


def extract_api_key(request: Request) -> str:
    """Extract API key from request headers (x-api-key or Authorization: Bearer)"""
    key = request.headers.get("x-api-key", "")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:]
    return key.strip()


def require_api_key(request: Request) -> dict:
    """FastAPI dependency: require valid API key"""
    api_key = extract_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    info = get_key_info(api_key)
    if info is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not info.get("enabled", True):
        raise HTTPException(status_code=403, detail="API key disabled")
    return {"key": api_key, **info}


def require_admin(request: Request):
    """FastAPI dependency: require admin key"""
    api_key = extract_api_key(request)
    data = load_keys()
    if api_key != data.get("admin_key"):
        raise HTTPException(status_code=403, detail="Admin key required")


def create_api_key(name: str) -> tuple[str, dict]:
    """Create a new API key. Returns (key_string, key_info)."""
    api_key = f"sk-{uuid.uuid4().hex[:24]}"
    data = load_keys()
    key_info = {
        "name": name,
        "enabled": True,
        "created": datetime.now(SGT).isoformat(),
        "usage": {
            "total_input": 0,
            "total_output": 0,
            "total_requests": 0,
            "last_used": None,
            "daily": {}
        }
    }
    data["keys"][api_key] = key_info
    save_keys(data)
    return api_key, key_info


def check_provider_access(key_info: dict, provider_id: str) -> bool:
    """Check if a key has access to a specific provider.

    If allowed_providers is not set, the key can access all providers.
    """
    allowed = key_info.get("allowed_providers")
    if not allowed:
        return True
    return provider_id in allowed


def get_accessible_providers(key_info: dict, all_providers: dict) -> dict:
    """Filter providers to only those this key can access."""
    allowed = key_info.get("allowed_providers")
    if not allowed:
        return all_providers
    return {k: v for k, v in all_providers.items() if k in allowed}
