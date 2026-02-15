"""Health check endpoint"""
from .config import get_known_models, get_providers


async def health():
    """Health check endpoint handler"""
    providers = get_providers()
    models = [m["id"] for m in get_known_models()]
    return {
        "status": "ok",
        "providers": list(providers.keys()),
        "models": models,
        "total_providers": len(providers),
        "total_models": len(models),
    }
