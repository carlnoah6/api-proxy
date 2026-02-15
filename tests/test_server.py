"""Unit tests for API Proxy server (provider-based architecture)"""
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
        "sk-test-all": {
            "name": "All Providers",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "allowed_providers": ["moonshot", "deepseek", "zhipu", "openai", "google"],
            "usage": {
                "total_input": 0, "total_output": 0, "total_requests": 0,
                "last_used": None, "daily": {}, "total_by_model": {}, "hourly": {}
            }
        },
        "sk-test-limited": {
            "name": "Limited User",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "allowed_providers": ["moonshot", "deepseek"],
            "usage": {
                "total_input": 0, "total_output": 0, "total_requests": 0,
                "last_used": None, "daily": {}, "total_by_model": {}, "hourly": {}
            }
        },
        "sk-test-unrestricted": {
            "name": "Unrestricted User",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "usage": {
                "total_input": 500, "total_output": 200, "total_requests": 10,
                "last_used": "2026-01-15T12:00:00+08:00",
                "daily": {
                    "2026-01-15": {
                        "input": 500, "output": 200, "requests": 10,
                        "by_model": {"kimi-k2.5": {"total": 700, "requests": 10}}
                    }
                },
                "total_by_model": {"kimi-k2.5": {"total": 700, "requests": 10}},
                "hourly": {
                    "2026-01-15 12:00": {
                        "by_model": {"kimi-k2.5": {"total": 700, "requests": 10}}
                    }
                }
            }
        },
        "sk-test-disabled": {
            "name": "Disabled User",
            "enabled": 0,
            "created": "2026-01-01T00:00:00+08:00",
            "usage": {
                "total_input": 0, "total_output": 0, "total_requests": 0,
                "last_used": None, "daily": {}, "total_by_model": {}, "hourly": {}
            }
        }
    }
}

TEST_MODELS = {
    "providers": {
        "moonshot": {
            "name": "Moonshot (Kimi)",
            "format": "openai",
            "base_url": "https://api.moonshot.cn/v1",
            "api_key_env": "KIMI_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["kimi", "moonshot"]
        },
        "deepseek": {
            "name": "DeepSeek",
            "format": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["deepseek"]
        },
        "zhipu": {
            "name": "ZAI / Zhipu (GLM)",
            "format": "openai",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "api_key_env": "ZAI_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["glm"]
        },
        "openai": {
            "name": "OpenAI",
            "format": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["gpt", "o1", "o3", "o4", "chatgpt"]
        },
        "google": {
            "name": "Google (Gemini)",
            "format": "openai",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key_env": "GOOGLE_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["gemini"]
        }
    },
    "known_models": [
        {"id": "kimi-k2.5", "provider": "moonshot", "owned_by": "moonshot"},
        {"id": "deepseek-chat", "provider": "deepseek", "owned_by": "deepseek"},
        {"id": "glm-5", "provider": "zhipu", "owned_by": "zhipu"},
        {"id": "gpt-4o", "provider": "openai", "owned_by": "openai"},
        {"id": "gemini-2.5-flash", "provider": "google", "owned_by": "google"}
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
    os.environ["KIMI_API_KEY"] = "sk-test-kimi"
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-deepseek"
    os.environ["ZAI_API_KEY"] = "sk-test-zai"
    os.environ["OPENAI_API_KEY"] = "sk-test-openai"
    os.environ["GOOGLE_API_KEY"] = "sk-test-google"

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
        assert data["admin_key"] == "sk-admin-test-key"
        assert "sk-test-all" in data["keys"]

    def test_load_keys_creates_default_when_missing(self, tmp_path):
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
        info = get_key_info("sk-test-all")
        assert info is not None
        assert info["name"] == "All Providers"

    def test_get_key_info_invalid(self):
        from src.auth import get_key_info
        assert get_key_info("sk-nonexistent") is None

    def test_get_key_info_disabled(self):
        from src.auth import get_key_info
        info = get_key_info("sk-test-disabled")
        assert info["enabled"] == 0

    def test_extract_api_key_from_header(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-test-key-123"}
        assert extract_api_key(request) == "sk-test-key-123"

    def test_extract_api_key_from_bearer(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {"authorization": "Bearer sk-test-bearer"}
        assert extract_api_key(request) == "sk-test-bearer"

    def test_extract_api_key_missing(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {}
        assert extract_api_key(request) == ""

    def test_extract_api_key_x_api_key_takes_precedence(self):
        from src.auth import extract_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-header", "authorization": "Bearer sk-bearer"}
        assert extract_api_key(request) == "sk-header"

    def test_require_api_key_valid(self):
        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {"x-api-key": "sk-test-all"}
        result = require_api_key(request)
        assert result["key"] == "sk-test-all"

    def test_require_api_key_missing_raises_401(self):
        from fastapi import HTTPException
        from src.auth import require_api_key
        request = MagicMock()
        request.headers = {}
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
        data = load_keys()
        assert key in data["keys"]

    def test_save_and_load_keys(self):
        from src.auth import load_keys, save_keys
        data = load_keys()
        data["keys"]["sk-new-test"] = {"name": "Saved", "enabled": True, "created": "2026-01-01",
                                        "usage": {"total_input": 0, "total_output": 0, "total_requests": 0}}
        save_keys(data)
        reloaded = load_keys()
        assert reloaded["keys"]["sk-new-test"]["name"] == "Saved"


# ══════════════════════════════════════════════
#  Provider ACL Tests
# ══════════════════════════════════════════════


class TestProviderACL:
    def test_check_provider_access_allowed(self):
        from src.auth import check_provider_access
        key_info = {"allowed_providers": ["moonshot", "deepseek"]}
        assert check_provider_access(key_info, "moonshot") is True
        assert check_provider_access(key_info, "deepseek") is True

    def test_check_provider_access_denied(self):
        from src.auth import check_provider_access
        key_info = {"allowed_providers": ["moonshot", "deepseek"]}
        assert check_provider_access(key_info, "openai") is False
        assert check_provider_access(key_info, "google") is False

    def test_check_provider_access_unrestricted(self):
        from src.auth import check_provider_access
        assert check_provider_access({}, "anything") is True
        assert check_provider_access({"name": "test"}, "openai") is True

    def test_get_accessible_providers_filtered(self):
        from src.auth import get_accessible_providers
        all_p = {"moonshot": {}, "deepseek": {}, "openai": {}, "google": {}}
        key_info = {"allowed_providers": ["moonshot", "deepseek"]}
        result = get_accessible_providers(key_info, all_p)
        assert set(result.keys()) == {"moonshot", "deepseek"}

    def test_get_accessible_providers_unrestricted(self):
        from src.auth import get_accessible_providers
        all_p = {"moonshot": {}, "deepseek": {}, "openai": {}}
        result = get_accessible_providers({}, all_p)
        assert set(result.keys()) == {"moonshot", "deepseek", "openai"}


# ══════════════════════════════════════════════
#  Config & Model Resolution Tests
# ══════════════════════════════════════════════


class TestModelConfig:
    def test_get_providers(self):
        from src.config import get_providers
        providers = get_providers()
        assert "moonshot" in providers
        assert "deepseek" in providers
        assert "zhipu" in providers
        assert "openai" in providers
        assert "google" in providers

    def test_get_providers_has_api_keys(self):
        from src.config import get_providers
        providers = get_providers()
        assert providers["moonshot"]["api_key"] == "sk-test-kimi"
        assert providers["deepseek"]["api_key"] == "sk-test-deepseek"

    def test_get_known_models(self):
        from src.config import get_known_models
        models = get_known_models()
        assert len(models) == 5
        ids = [m["id"] for m in models]
        assert "kimi-k2.5" in ids
        assert "glm-5" in ids

    def test_resolve_model_exact_match(self):
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("kimi-k2.5", providers)
        assert pid == "moonshot"
        assert p is not None

    def test_resolve_model_prefix_match(self):
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("gpt-4o-mini", providers)
        assert pid == "openai"
        p2, pid2 = resolve_model("glm-4.7", providers)
        assert pid2 == "zhipu"
        p3, pid3 = resolve_model("deepseek-reasoner", providers)
        assert pid3 == "deepseek"

    def test_resolve_model_not_found(self):
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("nonexistent-model", providers)
        assert p is None
        assert pid is None

    def test_load_models_config_missing_file(self, tmp_path):
        os.environ["MODELS_CONFIG"] = str(tmp_path / "nonexistent.json")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]
        from src.config import get_known_models, get_providers
        assert get_providers() == {}
        assert get_known_models() == []


# ══════════════════════════════════════════════
#  Usage Recording Tests
# ══════════════════════════════════════════════


class TestUsageRecording:
    def test_record_usage(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "test-model")
        data = load_keys()
        usage = data["keys"]["sk-test-all"]["usage"]
        assert usage["total_input"] == 100
        assert usage["total_output"] == 50
        assert usage["total_requests"] == 1

    def test_record_usage_accumulates(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "test-model")
        record_usage("sk-test-all", 200, 100, "test-model")
        data = load_keys()
        usage = data["keys"]["sk-test-all"]["usage"]
        assert usage["total_input"] == 300
        assert usage["total_output"] == 150
        assert usage["total_requests"] == 2

    def test_record_usage_by_model(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "model-a")
        record_usage("sk-test-all", 200, 100, "model-b")
        data = load_keys()
        by_model = data["keys"]["sk-test-all"]["usage"]["total_by_model"]
        assert "model-a" in by_model
        assert "model-b" in by_model

    def test_record_usage_nonexistent_key(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-nonexistent", 100, 50, "test-model")
        data = load_keys()
        assert "sk-nonexistent" not in data["keys"]

    def test_record_usage_daily_stats(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "test-model")
        data = load_keys()
        daily = data["keys"]["sk-test-all"]["usage"]["daily"]
        assert len(daily) > 0

    def test_record_usage_hourly_stats(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "test-model")
        data = load_keys()
        hourly = data["keys"]["sk-test-all"]["usage"]["hourly"]
        assert len(hourly) > 0

    def test_record_usage_model_total_aggregation(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "model-x")
        record_usage("sk-test-all", 200, 100, "model-x")
        data = load_keys()
        by_model = data["keys"]["sk-test-all"]["usage"]["total_by_model"]
        assert by_model["model-x"]["total"] == 450
        assert by_model["model-x"]["requests"] == 2


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
        assert "providers" in data
        assert "models" in data

    @pytest.mark.asyncio
    async def test_health_endpoint_counts(self):
        from fastapi.testclient import TestClient
        from src.app import app
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["total_providers"] == 5
        assert data["total_models"] == 5


# ══════════════════════════════════════════════
#  Admin Endpoint Tests
# ══════════════════════════════════════════════


class TestAdminEndpoints:
    def _client(self):
        from fastapi.testclient import TestClient
        from src.app import app
        return TestClient(app)

    def test_admin_list_keys(self):
        resp = self._client().get("/admin/keys", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        assert "keys" in resp.json()

    def test_admin_list_keys_requires_admin(self):
        resp = self._client().get("/admin/keys", headers={"x-api-key": "sk-test-all"})
        assert resp.status_code == 403

    def test_admin_create_key(self):
        resp = self._client().post("/admin/keys", headers={"x-api-key": "sk-admin-test-key"},
                                    json={"name": "NewUser"})
        assert resp.status_code == 200
        assert "api_key" in resp.json()

    def test_admin_disable_key(self):
        resp = self._client().post("/admin/keys/sk-test-all/disable",
                                    headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200

    def test_admin_enable_key(self):
        resp = self._client().post("/admin/keys/sk-test-disabled/enable",
                                    headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200

    def test_admin_delete_key(self):
        resp = self._client().delete("/admin/keys/sk-test-disabled/",
                                      headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200

    def test_admin_key_not_found(self):
        resp = self._client().post("/admin/keys/sk-does-not-exist/disable",
                                    headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 404

    def test_admin_key_usage(self):
        resp = self._client().get("/admin/keys/sk-test-unrestricted/usage",
                                   headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        assert "usage" in resp.json()

    def test_admin_total_usage(self):
        resp = self._client().get("/admin/usage", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "total_input_tokens" in data
        assert "total_keys" in data

    def test_admin_daily_usage(self):
        resp = self._client().get("/admin/usage/daily", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        assert "days" in resp.json()

    def test_admin_hourly_usage(self):
        resp = self._client().get("/admin/usage/hourly", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        assert "hours" in resp.json()

    def test_admin_models_status(self):
        resp = self._client().get("/admin/models", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "known_models" in data


# ══════════════════════════════════════════════
#  Proxy Auth Integration Tests
# ══════════════════════════════════════════════


class TestProxyAuth:
    def _client(self):
        from fastapi.testclient import TestClient
        from src.app import app
        return TestClient(app)

    def test_proxy_no_key_returns_401(self):
        resp = self._client().post("/v1/messages", json={"model": "test", "messages": []})
        assert resp.status_code == 401

    def test_proxy_invalid_key_returns_401(self):
        resp = self._client().post("/v1/messages", headers={"x-api-key": "sk-invalid"},
                                    json={"model": "test", "messages": []})
        assert resp.status_code == 401

    def test_proxy_disabled_key_returns_403(self):
        resp = self._client().post("/v1/messages", headers={"x-api-key": "sk-test-disabled"},
                                    json={"model": "test", "messages": []})
        assert resp.status_code == 403


# ══════════════════════════════════════════════
#  Model Routing + ACL Integration Tests
# ══════════════════════════════════════════════


class TestModelRouting:
    def _client(self):
        from fastapi.testclient import TestClient
        from src.app import app
        return TestClient(app)

    def test_unknown_model_returns_400(self):
        resp = self._client().post("/v1/chat/completions",
                                    headers={"x-api-key": "sk-test-all"},
                                    json={"model": "nonexistent-xyz", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 400
        assert "not available" in resp.json()["error"]["message"]

    def test_provider_access_denied(self):
        resp = self._client().post("/v1/chat/completions",
                                    headers={"x-api-key": "sk-test-limited"},
                                    json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 403
        assert "provider_access_denied" in json.dumps(resp.json())

    def test_provider_access_denied_prefix(self):
        """Prefix-matched model should also be denied if provider not allowed."""
        resp = self._client().post("/v1/chat/completions",
                                    headers={"x-api-key": "sk-test-limited"},
                                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 403

    def test_models_endpoint_filtered_by_provider(self):
        resp = self._client().get("/v1/models", headers={"x-api-key": "sk-test-limited"})
        assert resp.status_code == 200
        data = resp.json()
        model_ids = {m["id"] for m in data["data"]}
        assert "kimi-k2.5" in model_ids
        assert "deepseek-chat" in model_ids
        assert "gpt-4o" not in model_ids
        assert "gemini-2.5-flash" not in model_ids

    def test_models_endpoint_all_for_full_access(self):
        resp = self._client().get("/v1/models", headers={"x-api-key": "sk-test-all"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 5

    def test_models_endpoint_unrestricted_sees_all(self):
        resp = self._client().get("/v1/models", headers={"x-api-key": "sk-test-unrestricted"})
        data = resp.json()
        assert len(data["data"]) == 5

    def test_models_endpoint_requires_auth(self):
        resp = self._client().get("/v1/models")
        assert resp.status_code == 401

    def test_models_endpoint_includes_provider_field(self):
        resp = self._client().get("/v1/models", headers={"x-api-key": "sk-test-all"})
        data = resp.json()
        providers = {m["id"]: m["provider"] for m in data["data"]}
        assert providers["kimi-k2.5"] == "moonshot"
        assert providers["glm-5"] == "zhipu"
        assert providers["gpt-4o"] == "openai"
