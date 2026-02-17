"""Unit tests for API Proxy server (provider-based architecture with routing)"""
import json
import os
import sys
from pathlib import Path

import pytest

# ── Test data ──

TEST_KEYS = {
    "admin_key": "sk-admin-test-key",
    "keys": {
        "sk-test-all": {
            "name": "All Providers",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "allowed_providers": ["moonshot", "deepseek", "zhipu", "openai", "google", "aiberm"],
            "usage": {
                "total_input": 0, "total_output": 0, "total_requests": 0,
                "last_used": None, "daily": {}, "total_by_model": {}, "hourly": {}
            }
        },
        "sk-test-limited": {
            "name": "Limited User",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "allowed_providers": ["moonshot", "deepseek", "aiberm"],
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
        "openai": {
            "name": "OpenAI",
            "format": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": ["gpt", "o1", "o3", "o4"]
        },
        "aiberm": {
            "name": "Aiberm (Aggregator)",
            "format": "openai",
            "base_url": "https://aiberm.com/v1",
            "api_key_env": "AIBERM_API_KEY",
            "auth_header": "authorization_bearer",
            "chat_endpoint": "/chat/completions",
            "model_prefixes": []
        }
    },
    "routing": {
        "kimi": ["moonshot"],
        "deepseek": ["deepseek"],
        "gpt": ["openai", "aiberm"],
        "claude": ["aiberm"]
    },
    "known_models": [
        {"id": "kimi-k2.5", "provider": "moonshot", "owned_by": "moonshot"},
        {"id": "deepseek-chat", "provider": "deepseek", "owned_by": "deepseek"},
        {"id": "gpt-4o", "provider": "openai", "owned_by": "openai"},
        {"id": "claude-opus-4-6-thinking", "provider": "aiberm", "owned_by": "anthropic"}
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
    os.environ["OPENAI_API_KEY"] = "sk-test-openai"
    os.environ["AIBERM_API_KEY"] = "sk-test-aiberm"

    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src"):
            del sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).parent.parent))
    yield
    sys.path.pop(0)


# ══════════════════════════════════════════════
#  Config / Routing Tests
# ══════════════════════════════════════════════


class TestConfig:
    def test_get_providers(self):
        from src.config import get_providers
        providers = get_providers()
        assert "moonshot" in providers
        assert "aiberm" in providers
        assert providers["moonshot"]["api_key"] == "sk-test-kimi"

    def test_get_routing(self):
        from src.config import get_routing
        routing = get_routing()
        assert routing["gpt"] == ["openai", "aiberm"]
        assert routing["claude"] == ["aiberm"]

    def test_resolve_model_direct_match(self):
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("kimi-k2.5", providers)
        assert pid == "moonshot"

    def test_resolve_model_routing_chain_first(self):
        """gpt-4o should route to openai (first in chain)."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("gpt-4o", providers)
        assert pid == "openai"

    def test_resolve_model_routing_chain_fallback(self):
        """gpt-4o with only aiberm allowed should fallback to aiberm."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("gpt-4o", providers, allowed_providers={"aiberm", "moonshot"})
        assert pid == "aiberm"

    def test_resolve_model_routing_chain_no_access(self):
        """gpt-4o with no matching provider returns None."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("gpt-4o", providers, allowed_providers={"moonshot"})
        assert p is None
        assert pid is None

    def test_resolve_model_claude_only_aiberm(self):
        """claude-* only routes to aiberm."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("claude-opus-4-6-thinking", providers)
        assert pid == "aiberm"

    def test_resolve_model_unknown(self):
        from src.config import get_providers, resolve_model
        providers = get_providers()
        p, pid = resolve_model("nonexistent-model", providers)
        assert p is None

    def test_resolve_model_skips_no_api_key(self):
        """Provider without API key is skipped in routing chain."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        providers["openai"]["api_key"] = ""  # simulate missing key
        p, pid = resolve_model("gpt-4o", providers)
        assert pid == "aiberm"  # falls back

    def test_resolve_model_longest_prefix(self):
        """Longest prefix match wins."""
        from src.config import get_providers, resolve_model
        providers = get_providers()
        # "deepseek" prefix matches "deepseek-chat"
        p, pid = resolve_model("deepseek-chat", providers)
        assert pid == "deepseek"


# ══════════════════════════════════════════════
#  Key Authentication Tests
# ══════════════════════════════════════════════


class TestKeyAuth:
    def test_load_keys(self):
        from src.auth import load_keys
        data = load_keys()
        assert data["admin_key"] == "sk-admin-test-key"

    def test_load_keys_creates_default_when_missing(self, tmp_path):
        os.environ["KEYS_FILE"] = str(tmp_path / "nonexistent.json")
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]
        from src.auth import load_keys
        data = load_keys()
        assert "admin_key" in data
        assert "keys" in data

    def test_require_api_key_missing(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"model": "test"})
        assert resp.status_code == 401

    def test_require_api_key_invalid(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "invalid-key"},
            json={"model": "test"},
        )
        assert resp.status_code == 401

    def test_require_api_key_disabled(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "sk-test-disabled"},
            json={"model": "test"},
        )
        assert resp.status_code == 403


# ══════════════════════════════════════════════
#  Usage Recording Tests
# ══════════════════════════════════════════════


class TestUsageRecording:
    def test_record_usage_updates_totals(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        data = load_keys()
        usage = data["keys"]["sk-test-all"]["usage"]
        assert usage["total_input"] == 100
        assert usage["total_output"] == 50
        assert usage["total_requests"] == 1

    def test_record_usage_daily_breakdown(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        data = load_keys()
        daily = data["keys"]["sk-test-all"]["usage"]["daily"]
        assert len(daily) == 1
        day = list(daily.values())[0]
        assert day["input"] == 100
        assert day["output"] == 50
        assert day["requests"] == 1

    def test_record_usage_by_model(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        data = load_keys()
        by_model = data["keys"]["sk-test-all"]["usage"]["total_by_model"]
        assert "kimi-k2.5" in by_model
        assert by_model["kimi-k2.5"]["total"] == 150
        assert by_model["kimi-k2.5"]["requests"] == 1

    def test_record_usage_hourly(self):
        from src.auth import load_keys
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        data = load_keys()
        hourly = data["keys"]["sk-test-all"]["usage"]["hourly"]
        assert len(hourly) == 1

    def test_record_usage_nonexistent_key(self):
        from src.usage import record_usage
        record_usage("nonexistent-key", 100, 50, "test")  # should not raise


# ══════════════════════════════════════════════
#  Admin Endpoints Tests
# ══════════════════════════════════════════════


class TestAdminEndpoints:
    def _client(self):
        from fastapi.testclient import TestClient

        from src.app import app
        return TestClient(app)

    def test_admin_total_usage(self):
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        client = self._client()
        resp = client.get("/admin/usage", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_input_tokens"] >= 100

    def test_admin_list_keys(self):
        client = self._client()
        resp = client.get("/admin/keys", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["keys"]) == 4

    def test_admin_key_usage(self):
        from src.usage import record_usage
        record_usage("sk-test-all", 200, 100, "deepseek-chat")
        client = self._client()
        resp = client.get(
            "/admin/keys/sk-test-all/usage",
            headers={"x-api-key": "sk-admin-test-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["total_input"] == 200

    def test_admin_daily_usage(self):
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        client = self._client()
        resp = client.get(
            "/admin/usage/daily", headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["days"]) >= 1

    def test_admin_hourly_usage(self):
        from src.usage import record_usage
        record_usage("sk-test-all", 100, 50, "kimi-k2.5")
        client = self._client()
        resp = client.get(
            "/admin/usage/hourly", headers={"x-api-key": "sk-admin-test-key"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["hours"]) >= 1

    def test_admin_models_status(self):
        client = self._client()
        resp = client.get("/admin/models", headers={"x-api-key": "sk-admin-test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["providers"]) == 4


# ══════════════════════════════════════════════
#  Proxy Auth Tests
# ══════════════════════════════════════════════


class TestProxyAuth:
    def test_proxy_no_key_returns_401(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={"model": "kimi-k2.5"})
        assert resp.status_code == 401

    def test_proxy_invalid_key_returns_401(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "invalid"},
            json={"model": "kimi-k2.5"},
        )
        assert resp.status_code == 401

    def test_proxy_disabled_key_returns_403(self):
        from fastapi.testclient import TestClient

        from src.app import app
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            headers={"x-api-key": "sk-test-disabled"},
            json={"model": "kimi-k2.5"},
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
            "/v1/chat/completions",
            headers={"x-api-key": "sk-test-all"},
            json={"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400

    def test_routing_fallback_to_aiberm(self):
        """Limited user (no openai) requesting gpt-4o should resolve to aiberm."""
        from src.config import get_providers, resolve_model

        providers = get_providers()
        p, pid = resolve_model("gpt-4o", providers, allowed_providers={"moonshot", "deepseek", "aiberm"})
        assert pid == "aiberm"

    def test_claude_routes_to_aiberm(self):
        """Claude models should always route to aiberm."""
        from src.config import get_providers, resolve_model

        providers = get_providers()
        p, pid = resolve_model("claude-opus-4-6-thinking", providers)
        assert pid == "aiberm"
        # Limited user with aiberm access can also reach claude
        p2, pid2 = resolve_model("claude-opus-4-6-thinking", providers, allowed_providers={"moonshot", "deepseek", "aiberm"})
        assert pid2 == "aiberm"

    def test_models_endpoint_filtered_by_provider(self):
        client = self._client()
        resp = client.get("/v1/models", headers={"x-api-key": "sk-test-limited"})
        assert resp.status_code == 200
        data = resp.json()
        model_ids = {m["id"] for m in data["data"]}
        assert "kimi-k2.5" in model_ids
        assert "deepseek-chat" in model_ids
        assert "claude-opus-4-6-thinking" in model_ids  # via aiberm
        # gpt-4o should also be visible (reachable via aiberm fallback)
        assert "gpt-4o" in model_ids

    def test_models_endpoint_all_for_full_access(self):
        client = self._client()
        resp = client.get("/v1/models", headers={"x-api-key": "sk-test-all"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 4

    def test_models_endpoint_unrestricted_sees_all(self):
        client = self._client()
        resp = client.get("/v1/models", headers={"x-api-key": "sk-test-unrestricted"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 4

    def test_models_endpoint_requires_auth(self):
        client = self._client()
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_models_endpoint_includes_provider_field(self):
        client = self._client()
        resp = client.get("/v1/models", headers={"x-api-key": "sk-test-all"})
        data = resp.json()
        providers = {m["id"]: m["provider"] for m in data["data"]}
        assert providers["kimi-k2.5"] == "moonshot"
        assert providers["claude-opus-4-6-thinking"] == "aiberm"
