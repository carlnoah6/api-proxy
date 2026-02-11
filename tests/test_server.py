"""Unit tests for API Proxy server (modular structure)"""
import json
import os
import sys
from pathlib import Path

import pytest

# ── Test data ──

TEST_KEYS = {
    "admin_key": "sk-admin-test-key",
    "keys": {
        "sk-test-user1": {
            "name": "Test User 1",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "usage": {
                "total_input": 0,
                "total_output": 0,
                "total_requests": 0,
                "last_used": None,
                "daily": {},
                "total_by_model": {},
                "hourly": {}
            }
        },
        "sk-test-disabled": {
            "name": "Disabled User",
            "enabled": 0,
            "created": "2026-01-01T00:00:00+08:00",
            "usage": {
                "total_input": 0,
                "total_output": 0,
                "total_requests": 0,
                "last_used": None,
                "daily": {},
                "total_by_model": {},
                "hourly": {}
            }
        }
    }
}


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    """Set up test environment before importing modules"""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps(TEST_KEYS, indent=2))
    fb_file = tmp_path / "fallback.json"
    fb_file.write_text(json.dumps({
        "enabled": True,
        "health_poll_interval_seconds": 30,
        "min_remaining_fraction": 0.05,
        "tiers": [
            {"name": "Test Tier", "model": "test-model", "type": "antigravity", "health_key": "test-model"}
        ]
    }))

    os.environ["KEYS_FILE"] = str(keys_file)
    os.environ["FALLBACK_CONFIG"] = str(fb_file)
    os.environ["PROXY_PORT"] = "19999"

    # Clear cached modules so config reloads
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src"):
            del sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).parent.parent))
    yield
    sys.path.pop(0)


# ── Key Authentication Tests ──


class TestKeyAuth:
    def test_load_keys(self):
        from src.auth import load_keys
        data = load_keys()
        assert "admin_key" in data
        assert data["admin_key"] == "sk-admin-test-key"
        assert "sk-test-user1" in data["keys"]

    def test_get_key_info_valid(self):
        from src.auth import get_key_info
        info = get_key_info("sk-test-user1")
        assert info is not None
        assert info["name"] == "Test User 1"
        assert info["enabled"] == 1

    def test_get_key_info_invalid(self):
        from src.auth import get_key_info
        info = get_key_info("sk-nonexistent-key")
        assert info is None

    def test_get_key_info_disabled(self):
        from src.auth import get_key_info
        info = get_key_info("sk-test-disabled")
        assert info is not None
        assert info["enabled"] == 0


# ── Usage Recording Tests ──


class TestUsageRecording:
    def test_record_usage(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "test-model")
        data = load_keys()
        usage = data["keys"]["sk-test-user1"]["usage"]
        assert usage["total_input"] == 100
        assert usage["total_output"] == 50
        assert usage["total_requests"] == 1
        assert usage["last_used"] is not None

    def test_record_usage_accumulates(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "test-model")
        record_usage("sk-test-user1", 200, 100, "test-model")
        data = load_keys()
        usage = data["keys"]["sk-test-user1"]["usage"]
        assert usage["total_input"] == 300
        assert usage["total_output"] == 150
        assert usage["total_requests"] == 2

    def test_record_usage_by_model(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "model-a")
        record_usage("sk-test-user1", 200, 100, "model-b")
        data = load_keys()
        total_by_model = data["keys"]["sk-test-user1"]["usage"]["total_by_model"]
        assert "model-a" in total_by_model
        assert "model-b" in total_by_model
        assert total_by_model["model-a"]["requests"] == 1
        assert total_by_model["model-b"]["requests"] == 1

    def test_record_usage_nonexistent_key(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-nonexistent", 100, 50, "test-model")
        data = load_keys()
        assert "sk-nonexistent" not in data["keys"]


# ── Format Conversion Tests ──


class TestFormatConversion:
    def test_anthropic_to_openai_basic(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert result["model"] == "test-model"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["max_tokens"] == 100

    def test_anthropic_to_openai_with_system(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are helpful"

    def test_anthropic_to_openai_content_blocks(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert result["messages"][0]["content"] == "Hello\nWorld"

    def test_openai_to_anthropic_basic(self):
        from src.proxy import openai_to_anthropic
        resp_data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        result = openai_to_anthropic(resp_data, "test-model")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "test-model"
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello there!"


# ── Fallback Config Tests ──


class TestFallbackConfig:
    def test_load_fallback_config(self):
        from src.fallback import load_fallback_config
        config = load_fallback_config()
        assert config["enabled"] is True
        assert len(config["tiers"]) > 0
        assert config["tiers"][0]["name"] == "Test Tier"

    def test_health_cache_init(self):
        from src.fallback import HealthCache
        cache = HealthCache()
        assert cache.models == {}
        assert cache.last_poll == 0

    def test_health_cache_get_remaining_unknown(self):
        from src.fallback import HealthCache
        cache = HealthCache()
        remaining = cache.get_remaining("unknown-model")
        assert remaining == -1

    def test_health_cache_is_available_unknown(self):
        from src.fallback import HealthCache
        cache = HealthCache()
        available = cache.is_available("unknown-model")
        assert available is True


# ── Health Endpoint Test ──


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
