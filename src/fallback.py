"""Smart fallback logic - health monitoring and tier selection"""
import asyncio
import json
import time
from typing import Optional

import httpx

from .config import FALLBACK_CONFIG_FILE, UPSTREAM, log


def load_fallback_config() -> dict:
    """Load fallback configuration from file"""
    default = {
        "enabled": True,
        "health_poll_interval_seconds": 30,
        "min_remaining_fraction": 0.05,
        "tiers": [
            {
                "name": "Claude Opus 4.6",
                "model": "claude-opus-4-6-thinking",
                "type": "antigravity",
                "health_key": "claude-opus-4-6-thinking"
            },
            {
                "name": "Gemini 3 Pro High",
                "model": "gemini-3-pro-high",
                "type": "antigravity",
                "health_key": "gemini-3-pro-high"
            }
        ]
    }
    if FALLBACK_CONFIG_FILE.exists():
        try:
            return json.loads(FALLBACK_CONFIG_FILE.read_text())
        except Exception:
            pass
    # Write default config
    FALLBACK_CONFIG_FILE.write_text(json.dumps(default, indent=2, ensure_ascii=False))
    return default


class HealthCache:
    """Cache for upstream /health endpoint model quotas"""

    def __init__(self):
        self.models: dict = {}
        self.last_poll: float = 0
        self.poll_interval: int = 30
        self._lock = asyncio.Lock()

    async def poll(self, client: httpx.AsyncClient):
        """Poll /health to update quota cache"""
        now = time.time()
        if now - self.last_poll < self.poll_interval:
            return

        async with self._lock:
            # Double check after acquiring lock
            if time.time() - self.last_poll < self.poll_interval:
                return
            try:
                resp = await client.get(f"{UPSTREAM}/health", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    accounts = data.get("accounts", [])
                    # Aggregate models from ALL accounts (take max remaining)
                    aggregated_models = {}
                    for acc in accounts:
                        models = acc.get("models", {})
                        for model_name, info in models.items():
                            if model_name not in aggregated_models:
                                aggregated_models[model_name] = info
                            else:
                                current_rem = aggregated_models[model_name].get("remainingFraction", 0)
                                new_rem = info.get("remainingFraction", 0)
                                if new_rem > current_rem:
                                    aggregated_models[model_name] = info

                    self.models = aggregated_models
                    self.last_poll = time.time()
                    log.debug(f"Health poll: {len(self.models)} models updated from {len(accounts)} accounts")
            except Exception as e:
                log.warning(f"Health poll failed: {e}")

    def get_remaining(self, model_key: str) -> float:
        """Get remaining quota fraction (0.0-1.0), -1 for unknown"""
        info = self.models.get(model_key, {})
        return info.get("remainingFraction", -1)

    def is_available(self, model_key: str, min_remaining: float = 0.05) -> bool:
        """Check if a model has enough quota"""
        remaining = self.get_remaining(model_key)
        if remaining < 0:
            return True  # Unknown = assume available
        return remaining >= min_remaining


# Global health cache instance
health_cache = HealthCache()


async def pick_tier(client: httpx.AsyncClient, requested_model: str) -> Optional[dict]:
    """Select the best tier based on quota availability"""
    config = load_fallback_config()
    if not config.get("enabled"):
        return None

    await health_cache.poll(client)

    min_remaining = config.get("min_remaining_fraction", 0.05)
    tiers = config.get("tiers", [])

    # Find the tier matching the requested model
    requested_tier_idx = -1
    for i, tier in enumerate(tiers):
        if tier["model"] == requested_model:
            requested_tier_idx = i
            break

    # If requested model not in tiers, don't fallback
    if requested_tier_idx < 0:
        return None

    # Check if requested model has quota
    requested_tier = tiers[requested_tier_idx]
    if requested_tier["type"] in ("antigravity", "upstream"):
        if health_cache.is_available(requested_tier.get("health_key", requested_model), min_remaining):
            return None  # Original model is fine
    else:
        return None  # Non-upstream models don't have health check

    # Requested model is low → find fallback
    for i, tier in enumerate(tiers):
        if i == requested_tier_idx:
            continue

        if tier["type"] in ("antigravity", "upstream"):
            if health_cache.is_available(tier.get("health_key", tier["model"]), min_remaining):
                log.info(f"⚡ Fallback: {requested_model} → {tier['model']} (quota low)")
                return tier
        else:
            log.info(f"⚡ Fallback: {requested_model} → {tier['model']} (all upstream tiers exhausted)")
            return tier

    return None
