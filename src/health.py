"""Health check endpoint"""
from .config import UPSTREAM
from .fallback import load_fallback_config


async def health():
    """Health check endpoint handler"""
    config = load_fallback_config()
    return {
        "status": "ok",
        "upstream": UPSTREAM,
        "fallback_enabled": config.get("enabled", True),
        "tiers": len(config.get("tiers", []))
    }
