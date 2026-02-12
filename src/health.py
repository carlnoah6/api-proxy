"""Health check endpoint"""
from .config import get_models_registry


async def health():
    """Health check endpoint handler"""
    registry = get_models_registry()
    return {
        "status": "ok",
        "models": list(registry.keys()),
        "total_models": len(registry),
    }
