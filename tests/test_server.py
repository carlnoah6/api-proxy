"""Unit tests for API Proxy server (modular structure)"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        },
        "sk-test-user2": {
            "name": "Test User 2",
            "enabled": 1,
            "created": "2026-01-15T00:00:00+08:00",
            "usage": {
                "total_input": 500,
                "total_output": 200,
                "total_requests": 10,
                "last_used": "2026-01-15T12:00:00+08:00",
                "daily": {
                    "2026-01-15": {
                        "input": 500,
                        "output": 200,
                        "requests": 10,
                        "by_model": {
                            "claude-sonnet": {"total": 700, "requests": 10}
                        }
                    }
                },
                "total_by_model": {
                    "claude-sonnet": {"total": 700, "requests": 10}
                },
                "hourly": {
                    "2026-01-15 12:00": {
                        "by_model": {
                            "claude-sonnet": {"total": 700, "requests": 10}
                        }
                    }
                }
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
            {"name": "Test Tier", "model": "test-model", "type": "antigravity", "health_key": "test-model"},
            {"name": "Fallback Tier", "model": "fallback-model", "type": "antigravity", "health_key": "fallback-model"}
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


# ══════════════════════════════════════════════
#  Key Authentication Tests
# ══════════════════════════════════════════════


class TestKeyAuth:
    def test_load_keys(self):
        from src.auth import load_keys
        data = load_keys()
        assert "admin_key" in data
        assert data["admin_key"] == "sk-admin-test-key"
        assert "sk-test-user1" in data["keys"]

    def test_load_keys_creates_default_when_missing(self, tmp_path):
        """When keys file doesn't exist, returns default structure"""
        os.environ["KEYS_FILE"] = str(tmp_path / "nonexistent.json")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]
        from src.auth import load_keys
        data = load_keys()
        assert "admin_key" in data
        assert "keys" in data

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

    def test_extract_api_key_from_header(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-test-key-123"}
        assert extract_api_key(request) == "sk-test-key-123"

    def test_extract_api_key_from_bearer(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {"authorization": "Bearer sk-test-bearer-key"}
        assert extract_api_key(request) == "sk-test-bearer-key"

    def test_extract_api_key_missing(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {}
        assert extract_api_key(request) == ""

    def test_extract_api_key_x_api_key_takes_precedence(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {
            "x-api-key": "sk-from-header",
            "authorization": "Bearer sk-from-bearer"
        }
        assert extract_api_key(request) == "sk-from-header"

    def test_require_api_key_valid(self):
        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-test-user1"}
        result = require_api_key(request)
        assert result["key"] == "sk-test-user1"
        assert result["name"] == "Test User 1"

    def test_require_api_key_missing_raises_401(self):
        from fastapi import HTTPException

        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(request)
        assert exc_info.value.status_code == 401

    def test_require_api_key_invalid_raises_401(self):
        from fastapi import HTTPException

        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-does-not-exist"}
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(request)
        assert exc_info.value.status_code == 401

    def test_require_api_key_disabled_raises_403(self):
        from fastapi import HTTPException

        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-test-disabled"}
        with pytest.raises(HTTPException) as exc_info:
            require_api_key(request)
        assert exc_info.value.status_code == 403

    def test_require_admin_valid(self):
        from src.auth import require_admin
        request = MagicMock()
        request.headers = {"x-api-key": "sk-admin-test-key"}
        # Should not raise
        require_admin(request)

    def test_require_admin_invalid_raises_403(self):
        from fastapi import HTTPException

        from src.auth import require_admin
        request = MagicMock()
        request.headers = {"x-api-key": "sk-not-admin"}
        with pytest.raises(HTTPException) as exc_info:
            require_admin(request)
        assert exc_info.value.status_code == 403

    def test_create_api_key(self):
        from src.auth import create_api_key, load_keys
        key, info = create_api_key("New User")
        assert key.startswith("sk-")
        assert info["name"] == "New User"
        assert info["enabled"] is True
        # Verify persisted
        data = load_keys()
        assert key in data["keys"]

    def test_save_and_load_keys(self):
        from src.auth import load_keys, save_keys
        data = load_keys()
        data["keys"]["sk-new-test"] = {
            "name": "Saved User",
            "enabled": True,
            "created": "2026-01-01",
            "usage": {"total_input": 0, "total_output": 0, "total_requests": 0}
        }
        save_keys(data)
        reloaded = load_keys()
        assert "sk-new-test" in reloaded["keys"]
        assert reloaded["keys"]["sk-new-test"]["name"] == "Saved User"


# ══════════════════════════════════════════════
#  Usage Recording Tests
# ══════════════════════════════════════════════


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

    def test_record_usage_daily_stats(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "test-model")
        data = load_keys()
        daily = data["keys"]["sk-test-user1"]["usage"]["daily"]
        assert len(daily) > 0
        today = list(daily.keys())[0]
        assert daily[today]["input"] == 100
        assert daily[today]["output"] == 50
        assert daily[today]["requests"] == 1

    def test_record_usage_hourly_stats(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "test-model")
        data = load_keys()
        hourly = data["keys"]["sk-test-user1"]["usage"]["hourly"]
        assert len(hourly) > 0
        hour_key = list(hourly.keys())[0]
        assert "by_model" in hourly[hour_key]
        assert "test-model" in hourly[hour_key]["by_model"]

    def test_record_usage_model_total_aggregation(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-user1", 100, 50, "model-x")
        record_usage("sk-test-user1", 200, 100, "model-x")
        data = load_keys()
        by_model = data["keys"]["sk-test-user1"]["usage"]["total_by_model"]
        assert by_model["model-x"]["total"] == 450  # (100+50) + (200+100)
        assert by_model["model-x"]["requests"] == 2


# ══════════════════════════════════════════════
#  Format Conversion Tests
# ══════════════════════════════════════════════


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

    def test_anthropic_to_openai_with_system_string(self):
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

    def test_anthropic_to_openai_with_system_list(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "system": [{"type": "text", "text": "System prompt A"}, {"type": "text", "text": "System prompt B"}],
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert result["messages"][0]["role"] == "system"
        assert "System prompt A" in result["messages"][0]["content"]
        assert "System prompt B" in result["messages"][0]["content"]

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

    def test_anthropic_to_openai_with_thinking_blocks(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Answer"}
                ]
            }],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        # Thinking blocks should be skipped
        assert result["messages"][0]["content"] == "Answer"

    def test_anthropic_to_openai_with_tool_use(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "search", "input": {"query": "hello"}}
                ]
            }],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert "search" in result["messages"][0]["content"]

    def test_anthropic_to_openai_with_image_block(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc"}},
                    {"type": "text", "text": "What is this?"}
                ]
            }],
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert "[Image]" in result["messages"][0]["content"]

    def test_anthropic_to_openai_stream_param(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": 100
        }
        result = anthropic_to_openai(body)
        assert result["stream"] is True

    def test_anthropic_to_openai_temperature_and_top_p(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
        result = anthropic_to_openai(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_anthropic_to_openai_stop_sequences(self):
        from src.proxy import anthropic_to_openai
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["STOP", "END"]
        }
        result = anthropic_to_openai(body)
        assert result["stop"] == ["STOP", "END"]

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

    def test_openai_to_anthropic_stop_reason_mapping(self):
        from src.proxy import openai_to_anthropic
        for openai_reason, anthropic_reason in [("stop", "end_turn"), ("length", "max_tokens")]:
            resp_data = {
                "choices": [{"message": {"content": "Hi"}, "finish_reason": openai_reason}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}
            }
            result = openai_to_anthropic(resp_data, "test-model")
            assert result["stop_reason"] == anthropic_reason

    def test_openai_to_anthropic_with_reasoning(self):
        from src.proxy import openai_to_anthropic
        resp_data = {
            "choices": [{
                "message": {"content": "Answer", "reasoning_content": "Let me think..."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        result = openai_to_anthropic(resp_data, "test-model")
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "Let me think..."
        assert result["content"][1]["type"] == "text"

    def test_openai_to_anthropic_empty_choices(self):
        from src.proxy import openai_to_anthropic
        resp_data = {"choices": [], "usage": {}}
        result = openai_to_anthropic(resp_data, "test-model")
        assert result["content"][0]["text"] == ""
        assert result["stop_reason"] == "end_turn"

    def test_openai_to_anthropic_usage_mapping(self):
        from src.proxy import openai_to_anthropic
        resp_data = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 456}
        }
        result = openai_to_anthropic(resp_data, "test-model")
        assert result["usage"]["input_tokens"] == 123
        assert result["usage"]["output_tokens"] == 456

    def test_openai_to_anthropic_message_id_format(self):
        from src.proxy import openai_to_anthropic
        resp_data = {
            "id": "chatcmpl-abc123",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {}
        }
        result = openai_to_anthropic(resp_data, "test-model")
        assert result["id"].startswith("msg_")


# ══════════════════════════════════════════════
#  Fallback Config Tests
# ══════════════════════════════════════════════


class TestFallbackConfig:
    def test_load_fallback_config(self):
        from src.fallback import load_fallback_config
        config = load_fallback_config()
        assert config["enabled"] is True
        assert len(config["tiers"]) > 0
        assert config["tiers"][0]["name"] == "Test Tier"

    def test_load_fallback_config_missing_file(self, tmp_path):
        """When config file doesn't exist, writes and returns default"""
        os.environ["FALLBACK_CONFIG"] = str(tmp_path / "nonexistent_fb.json")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]
        from src.fallback import load_fallback_config
        config = load_fallback_config()
        assert "enabled" in config
        assert "tiers" in config
        assert len(config["tiers"]) > 0

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
        assert available is True  # Unknown = assume available

    def test_health_cache_is_available_with_data(self):
        from src.fallback import HealthCache
        cache = HealthCache()
        cache.models = {
            "model-good": {"remainingFraction": 0.8},
            "model-low": {"remainingFraction": 0.01},
        }
        assert cache.is_available("model-good", 0.05) is True
        assert cache.is_available("model-low", 0.05) is False

    def test_health_cache_get_remaining_with_data(self):
        from src.fallback import HealthCache
        cache = HealthCache()
        cache.models = {"my-model": {"remainingFraction": 0.42}}
        assert cache.get_remaining("my-model") == 0.42

    @pytest.mark.asyncio
    async def test_health_cache_poll_skips_when_recent(self):
        """Poll should skip if last poll was recent"""
        import time

        from src.fallback import HealthCache
        cache = HealthCache()
        cache.last_poll = time.time()  # Just polled
        mock_client = AsyncMock()
        await cache.poll(mock_client)
        mock_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_health_cache_poll_updates_models(self):
        """Poll should update model data from upstream"""
        from src.fallback import HealthCache
        cache = HealthCache()
        cache.last_poll = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "accounts": [{
                "models": {
                    "claude-opus": {"remainingFraction": 0.75},
                    "gemini-pro": {"remainingFraction": 0.30}
                }
            }]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        await cache.poll(mock_client)
        assert cache.get_remaining("claude-opus") == 0.75
        assert cache.get_remaining("gemini-pro") == 0.30

    @pytest.mark.asyncio
    async def test_health_cache_poll_aggregates_multiple_accounts(self):
        """Poll should take max remaining across multiple accounts"""
        from src.fallback import HealthCache
        cache = HealthCache()
        cache.last_poll = 0

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "accounts": [
                {"models": {"claude": {"remainingFraction": 0.1}}},
                {"models": {"claude": {"remainingFraction": 0.9}}}
            ]
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        await cache.poll(mock_client)
        assert cache.get_remaining("claude") == 0.9  # Max of 0.1 and 0.9

    @pytest.mark.asyncio
    async def test_pick_tier_returns_none_when_disabled(self):
        """pick_tier returns None when fallback is disabled"""
        from src.fallback import pick_tier
        with patch("src.fallback.load_fallback_config", return_value={"enabled": False}):
            result = await pick_tier(AsyncMock(), "test-model")
            assert result is None

    @pytest.mark.asyncio
    async def test_pick_tier_returns_none_for_unknown_model(self):
        """pick_tier returns None for models not in tiers"""
        from src.fallback import pick_tier
        mock_client = AsyncMock()
        result = await pick_tier(mock_client, "unknown-model-xyz")
        assert result is None


# ══════════════════════════════════════════════
#  Health Endpoint Test
# ══════════════════════════════════════════════


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

    @pytest.mark.asyncio
    async def test_health_endpoint_includes_upstream(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "upstream" in data
        assert "fallback_enabled" in data
        assert "tiers" in data


# ══════════════════════════════════════════════
#  Admin Endpoint Tests (via TestClient)
# ══════════════════════════════════════════════


class TestAdminEndpoints:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.app import app
        return TestClient(app)

    def test_admin_list_keys(self):
        client = self._client()
        resp = client.get("/admin/keys", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "keys" in data
        assert len(data["keys"]) >= 2

    def test_admin_list_keys_requires_admin(self):
        client = self._client()
        resp = client.get("/admin/keys", headers={"x-api-key": "sk-test-user1"})
        assert resp.status_code == 403

    def test_admin_create_key(self):
        client = self._client()
        resp = client.post(
            "/admin/keys",
            headers={"x-api-key": "sk-admin-test-key"},
            json={"name": "NewTestUser"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "api_key" in data
        assert data["name"] == "NewTestUser"

    def test_admin_disable_key(self):
        client = self._client()
        resp = client.post(
            "/admin/keys/sk-test-user1/disable",
            headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "disabled"

    def test_admin_enable_key(self):
        client = self._client()
        # First disable, then enable
        client.post("/admin/keys/sk-test-disabled/enable",
                     headers={"x-api-key": "sk-admin-test-key"})
        resp = client.post(
            "/admin/keys/sk-test-disabled/enable",
            headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "enabled"

    def test_admin_delete_key(self):
        client = self._client()
        resp = client.delete(
            "/admin/keys/sk-test-disabled/",
            headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_admin_key_not_found(self):
        client = self._client()
        resp = client.post(
            "/admin/keys/sk-does-not-exist/disable",
            headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 404

    def test_admin_key_usage(self):
        client = self._client()
        resp = client.get(
            "/admin/keys/sk-test-user2/usage",
            headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test User 2"
        assert "usage" in data

    def test_admin_total_usage(self):
        client = self._client()
        resp = client.get("/admin/usage", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_input_tokens" in data
        assert "total_output_tokens" in data
        assert "total_requests" in data
        assert "total_keys" in data
        assert data["total_keys"] >= 2

    def test_admin_daily_usage(self):
        client = self._client()
        resp = client.get("/admin/usage/daily", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "days" in data

    def test_admin_hourly_usage(self):
        client = self._client()
        resp = client.get("/admin/usage/hourly", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "hours" in data

    def test_admin_fallback_status(self):
        from fastapi.testclient import TestClient

        from src.app import app
        # The fallback endpoint needs app.state.client for health polling
        with TestClient(app) as client:
            resp = client.get("/admin/fallback", headers={"x-api-key": "sk-admin-test-key"})
            assert resp.status_code == 200
            data = resp.json()
            assert "enabled" in data
            assert "tiers" in data


# ══════════════════════════════════════════════
#  Proxy Auth Integration Tests (via TestClient)
# ══════════════════════════════════════════════


class TestProxyAuth:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.app import app
        return TestClient(app)

    def test_proxy_no_key_returns_401(self):
        client = self._client()
        resp = client.post("/v1/messages", json={"model": "test", "messages": []})
        assert resp.status_code == 401

    def test_proxy_invalid_key_returns_401(self):
        client = self._client()
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-invalid-key"},
            json={"model": "test", "messages": []}
        )
        assert resp.status_code == 401

    def test_proxy_disabled_key_returns_403(self):
        client = self._client()
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-test-disabled"},
            json={"model": "test", "messages": []}
        )
        assert resp.status_code == 403
