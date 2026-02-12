"""Unit tests for API Proxy server (modular structure)"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

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

TEST_MODELS = {
    "models": [
        {
            "id": "claude-opus-4-6-thinking",
            "name": "Claude Opus 4.6 Thinking",
            "format": "anthropic",
            "base_url": "https://example.com/api",
            "api_key_env": "CLAUDE_API_KEY",
            "auth_header": "x-api-key",
            "chat_endpoint": "/v1/messages",
            "owned_by": "anthropic"
        },
        {
            "id": "deepseek-chat",
            "name": "DeepSeek Chat",
            "format": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "owned_by": "deepseek"
        },
        {
            "id": "kimi-k2.5",
            "name": "Kimi K2.5",
            "format": "openai",
            "base_url": "https://api.moonshot.cn/v1",
            "api_key_env": "KIMI_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "owned_by": "moonshot"
        }
    ]
}


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    """Set up test environment before importing modules"""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps(TEST_KEYS, indent=2))
    models_file = tmp_path / "models.json"
    models_file.write_text(json.dumps(TEST_MODELS, indent=2))

    os.environ["KEYS_FILE"] = str(keys_file)
    os.environ["MODELS_CONFIG"] = str(models_file)
    os.environ["PROXY_PORT"] = "19999"
    os.environ["CLAUDE_API_KEY"] = "sk-test-claude"
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek"
    os.environ["KIMI_API_KEY"] = "sk-test-kimi"

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
#  Config & Model Registry Tests
# ══════════════════════════════════════════════


class TestModelConfig:
    def test_load_models_config(self):
        from src.config import load_models_config
        models = load_models_config()
        assert len(models) == 3
        assert models[0]["id"] == "claude-opus-4-6-thinking"
        assert models[0]["format"] == "anthropic"
        assert models[1]["id"] == "deepseek-chat"
        assert models[1]["format"] == "openai"
        assert models[2]["id"] == "kimi-k2.5"
        assert models[2]["format"] == "openai"

    def test_get_models_registry(self):
        from src.config import get_models_registry
        registry = get_models_registry()
        assert "claude-opus-4-6-thinking" in registry
        assert "deepseek-chat" in registry
        assert "kimi-k2.5" in registry
        assert registry["claude-opus-4-6-thinking"]["format"] == "anthropic"

    def test_api_keys_resolved(self):
        from src.config import load_models_config
        models = load_models_config()
        assert models[0]["api_key"] == "sk-test-claude"
        assert models[1]["api_key"] == "sk-test-deepseek"
        assert models[2]["api_key"] == "sk-test-kimi"

    def test_load_models_config_missing_file(self, tmp_path):
        """When models config file doesn't exist, returns empty list"""
        os.environ["MODELS_CONFIG"] = str(tmp_path / "nonexistent_models.json")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]
        from src.config import load_models_config
        models = load_models_config()
        assert models == []


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
    async def test_health_endpoint_includes_models(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "models" in data
        assert "total_models" in data
        assert data["total_models"] == 3


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

    def test_admin_models_status(self):
        client = self._client()
        resp = client.get("/admin/models", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert data["total"] == 3
        model_ids = [m["id"] for m in data["models"]]
        assert "claude-opus-4-6-thinking" in model_ids
        assert "deepseek-chat" in model_ids
        assert "kimi-k2.5" in model_ids


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


# ══════════════════════════════════════════════
#  Model Routing Tests
# ══════════════════════════════════════════════


class TestModelRouting:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.app import app
        return TestClient(app)

    def test_unknown_model_returns_400(self):
        client = self._client()
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-test-user1"},
            json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        )
        assert resp.status_code == 400
        assert "not available" in resp.json()["error"]["message"]

    def test_wrong_endpoint_anthropic_model_on_openai(self):
        """Anthropic model on /v1/chat/completions should return 400."""
        client = self._client()
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "sk-test-user1"},
            json={"model": "claude-opus-4-6-thinking", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp.status_code == 400
        assert "wrong_endpoint" in json.dumps(resp.json())

    def test_wrong_endpoint_openai_model_on_anthropic(self):
        """OpenAI model on /v1/messages should return 400."""
        client = self._client()
        resp = client.post(
            "/v1/messages",
            headers={"x-api-key": "sk-test-user1"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        )
        assert resp.status_code == 400
        assert "Use /v1/chat/completions" in resp.json()["error"]["message"]

    def test_models_endpoint_lists_all_models(self):
        client = self._client()
        resp = client.get(
            "/v1/models",
            headers={"x-api-key": "sk-test-user1"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 3
        model_ids = {m["id"] for m in data["data"]}
        assert "claude-opus-4-6-thinking" in model_ids
        assert "deepseek-chat" in model_ids
        assert "kimi-k2.5" in model_ids

    def test_models_endpoint_includes_format(self):
        client = self._client()
        resp = client.get(
            "/v1/models",
            headers={"x-api-key": "sk-test-user1"}
        )
        data = resp.json()
        formats = {m["id"]: m["format"] for m in data["data"]}
        assert formats["claude-opus-4-6-thinking"] == "anthropic"
        assert formats["deepseek-chat"] == "openai"
        assert formats["kimi-k2.5"] == "openai"

    def test_models_endpoint_requires_auth(self):
        client = self._client()
        resp = client.get("/v1/models")
        assert resp.status_code == 401
