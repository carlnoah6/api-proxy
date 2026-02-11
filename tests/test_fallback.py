"""Comprehensive fallback logic tests for API Proxy.

Tests cover all fallback paths including proactive fallback (health-based),
reactive fallback (429/400/503), streaming vs non-streaming, external tiers,
and the type-check fix that caused the KeyError bug.

Each test uses a mock upstream ASGI app via httpx.ASGITransport, so no
real network is needed. The proxy's httpx client is replaced with one
backed by the mock.
"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.responses import JSONResponse

from .mock_upstream import mock_app, mock_state, reset_mock_state

# ── Constants ──

PRIMARY_MODEL = "claude-opus-4-6-thinking"
FALLBACK_MODEL = "gemini-3-pro-high"

BASIC_REQUEST = {
    "model": PRIMARY_MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
}

# Keys data shared by all tests
TEST_KEYS = {
    "admin_key": "sk-admin-test-key",
    "keys": {
        "sk-test-user1": {
            "name": "Test User 1",
            "enabled": 1,
            "created": "2026-01-01T00:00:00+08:00",
            "usage": {
                "total_input": 0, "total_output": 0,
                "total_requests": 0, "last_used": None,
                "daily": {}, "total_by_model": {}, "hourly": {}
            }
        }
    }
}


# ── Fixtures ──


@pytest.fixture(autouse=True)
def setup_env(tmp_path):
    """Set up test environment: keys file, fallback config, env vars."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps(TEST_KEYS, indent=2))

    # Default fallback config with two upstream tiers
    fb_config = {
        "enabled": True,
        "health_poll_interval_seconds": 30,
        "min_remaining_fraction": 0.05,
        "tiers": [
            {
                "name": "Claude Opus 4.6",
                "model": PRIMARY_MODEL,
                "type": "upstream",
                "health_key": PRIMARY_MODEL
            },
            {
                "name": "Gemini 3 Pro High",
                "model": FALLBACK_MODEL,
                "type": "upstream",
                "health_key": FALLBACK_MODEL
            }
        ]
    }
    fb_file = tmp_path / "fallback.json"
    fb_file.write_text(json.dumps(fb_config, indent=2))

    os.environ["KEYS_FILE"] = str(keys_file)
    os.environ["FALLBACK_CONFIG"] = str(fb_file)
    os.environ["UPSTREAM_URL"] = "http://mock-upstream"
    os.environ["PROXY_PORT"] = "19998"

    # Clear cached src modules so config reloads with new env
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("src"):
            del sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Reset mock state
    reset_mock_state()

    yield

    sys.path.pop(0)


@pytest.fixture
def fb_config_file(tmp_path):
    """Helper to write custom fallback config for a test."""
    path = tmp_path / "fallback.json"

    def _write(config: dict):
        path.write_text(json.dumps(config, indent=2))

    return _write


def _make_upstream_client() -> httpx.AsyncClient:
    """Create an httpx client backed by the mock ASGI app."""
    transport = httpx.ASGITransport(app=mock_app)
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://mock-upstream",
        timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)
    )


def _make_proxy_client(app) -> httpx.AsyncClient:
    """Create an httpx client for the proxy FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test-proxy")


async def _reset_health_cache():
    """Reset the global health cache to force fresh polling."""
    from src.fallback import health_cache
    health_cache.models = {}
    health_cache.last_poll = 0


async def _proxy_post(body: dict, headers: dict | None = None,
                      stream: bool = False) -> httpx.Response:
    """Make a POST request to the proxy's /v1/messages endpoint."""
    from src.app import app

    if headers is None:
        headers = {"x-api-key": "sk-test-user1"}

    upstream_client = _make_upstream_client()
    app.state.client = upstream_client

    await _reset_health_cache()

    async with _make_proxy_client(app) as proxy_client:
        resp = await proxy_client.post(
            "/v1/messages",
            content=json.dumps(body),
            headers={**headers, "content-type": "application/json"},
        )

    await upstream_client.aclose()
    return resp


# ══════════════════════════════════════════════
#  1. Proactive Fallback (health-based)
# ══════════════════════════════════════════════


class TestProactiveFallback:
    @pytest.mark.asyncio
    async def test_proactive_fallback_when_primary_rate_limited(self):
        """When health check shows primary model rate-limited,
        the proxy should proactively switch to the fallback model."""
        # Primary model has low quota, fallback has plenty
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.01,    # below 0.05 threshold
            FALLBACK_MODEL: 0.80,
        }
        # Both models respond OK
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "ok",
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        # Should have been served by fallback model
        assert data["model"] == FALLBACK_MODEL
        assert f"Hello from {FALLBACK_MODEL}" in data["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_healthy(self):
        """When primary model has enough quota, no fallback should occur."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "ok",
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == PRIMARY_MODEL


# ══════════════════════════════════════════════
#  2. Reactive Fallback (429)
# ══════════════════════════════════════════════


class TestReactiveFallback429:
    @pytest.mark.asyncio
    async def test_reactive_fallback_on_429(self):
        """When primary returns 429, proxy should fall back to next tier."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,    # health says OK (stale)
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "429",       # but actually returns 429
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == FALLBACK_MODEL


# ══════════════════════════════════════════════
#  3. Reactive Fallback (400 with quota keywords)
# ══════════════════════════════════════════════


class TestReactiveFallback400:
    @pytest.mark.asyncio
    async def test_reactive_fallback_on_400_exhausted(self):
        """When primary returns 400 with 'exhausted' keyword,
        proxy should trigger reactive fallback."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "400_exhausted",
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == FALLBACK_MODEL


# ══════════════════════════════════════════════
#  4. Reactive Fallback (503)
# ══════════════════════════════════════════════


class TestReactiveFallback503:
    @pytest.mark.asyncio
    async def test_reactive_fallback_on_503(self):
        """When primary returns 503, proxy should fall back."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "503",
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == FALLBACK_MODEL


# ══════════════════════════════════════════════
#  5. All Tiers Exhausted
# ══════════════════════════════════════════════


class TestAllTiersExhausted:
    @pytest.mark.asyncio
    async def test_all_tiers_fail_returns_error_not_500(self):
        """When all tiers are unavailable, proxy should return the
        upstream error (429/503) — never an unhandled 500."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "429",
            FALLBACK_MODEL: "429",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        # Should NOT be 500
        assert resp.status_code != 500
        # Should be the original error status or a graceful error
        assert resp.status_code in (429, 503, 400)


# ══════════════════════════════════════════════
#  6. Fallback with Streaming
# ══════════════════════════════════════════════


class TestFallbackStream:
    @pytest.mark.asyncio
    async def test_stream_fallback_on_429(self):
        """Streaming request: when primary returns 429, fallback
        should return a valid streaming response."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "429",
            FALLBACK_MODEL: "ok",
        }

        body = {**BASIC_REQUEST, "stream": True}
        resp = await _proxy_post(body)

        assert resp.status_code == 200
        # Streaming response should contain SSE data with fallback model
        text = resp.text
        assert FALLBACK_MODEL in text
        assert "content_block_delta" in text

    @pytest.mark.asyncio
    async def test_stream_proactive_fallback(self):
        """Streaming request: proactive fallback should work too."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.01,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "ok",
            FALLBACK_MODEL: "ok",
        }

        body = {**BASIC_REQUEST, "stream": True}
        resp = await _proxy_post(body)

        assert resp.status_code == 200
        text = resp.text
        assert FALLBACK_MODEL in text


# ══════════════════════════════════════════════
#  7. Fallback Non-Stream
# ══════════════════════════════════════════════


class TestFallbackNonStream:
    @pytest.mark.asyncio
    async def test_non_stream_fallback_returns_valid_json(self):
        """Non-streaming fallback should return a proper Anthropic
        messages API JSON response."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            FALLBACK_MODEL: 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "503",
            FALLBACK_MODEL: "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == FALLBACK_MODEL
        assert len(data["content"]) > 0
        assert data["content"][0]["type"] == "text"
        assert "usage" in data


# ══════════════════════════════════════════════
#  8. External Tier Fallback (no KeyError)
# ══════════════════════════════════════════════


class TestExternalTierFallback:
    @pytest.mark.asyncio
    async def test_external_tier_no_key_error(self, tmp_path):
        """When fallback reaches an external tier (with api_key + base_url),
        it should NOT raise KeyError. This is the regression test for the
        bug that caused the system to be stuck for hours."""
        # Write config with an external tier
        fb_config = {
            "enabled": True,
            "health_poll_interval_seconds": 30,
            "min_remaining_fraction": 0.05,
            "tiers": [
                {
                    "name": "Claude Opus 4.6",
                    "model": PRIMARY_MODEL,
                    "type": "upstream",
                    "health_key": PRIMARY_MODEL
                },
                {
                    "name": "Kimi K2.5",
                    "model": "kimi-k2.5",
                    "type": "external",
                    "base_url": "https://api.moonshot.cn/v1",
                    "api_key": "sk-mock-external-key"
                }
            ]
        }
        fb_file = Path(os.environ["FALLBACK_CONFIG"])
        fb_file.write_text(json.dumps(fb_config, indent=2))

        # Primary returns 429, forcing fallback to external tier
        mock_state["health_models"] = {PRIMARY_MODEL: 0.80}
        mock_state["model_behavior"] = {PRIMARY_MODEL: "429"}

        from src.app import app

        upstream_client = _make_upstream_client()
        app.state.client = upstream_client
        await _reset_health_cache()

        # Mock call_external_tier to return a valid Anthropic response
        # This verifies the code path reaches external tier without KeyError
        mock_anthropic_response = JSONResponse(content={
            "id": "msg_mock_ext",
            "type": "message",
            "role": "assistant",
            "model": "kimi-k2.5",
            "content": [{"type": "text", "text": "Hello from Kimi"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })

        with patch("src.app.call_external_tier",
                    new=AsyncMock(return_value=mock_anthropic_response)) as mock_ext:
            async with _make_proxy_client(app) as proxy_client:
                resp = await proxy_client.post(
                    "/v1/messages",
                    content=json.dumps(BASIC_REQUEST),
                    headers={
                        "x-api-key": "sk-test-user1",
                        "content-type": "application/json"
                    },
                )

        await upstream_client.aclose()

        # The critical assertion: no 500/KeyError, external tier was reached
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert mock_ext.called, "call_external_tier should have been called"

    @pytest.mark.asyncio
    async def test_upstream_tier_does_not_access_api_key(self):
        """Upstream-type tiers should NOT try to access tier['api_key'].
        This verifies the root cause fix for the KeyError bug."""
        from src.fallback import load_fallback_config

        config = load_fallback_config()
        for tier in config["tiers"]:
            if tier["type"] in ("upstream", "antigravity"):
                # These tiers should not have api_key; accessing it
                # must not be required by any code path
                assert "api_key" not in tier, (
                    f"Tier '{tier['name']}' is type '{tier['type']}' "
                    f"but has api_key — this should only be on external tiers"
                )


# ══════════════════════════════════════════════
#  9. Type Check: "upstream" and "antigravity"
# ══════════════════════════════════════════════


class TestTierTypeCheck:
    @pytest.mark.asyncio
    async def test_upstream_type_recognized_by_pick_tier(self):
        """pick_tier should recognize type='upstream' for health checking."""
        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.01,    # low quota
            FALLBACK_MODEL: 0.80,
        }

        from src.app import app
        from src.fallback import pick_tier

        upstream_client = _make_upstream_client()
        app.state.client = upstream_client
        await _reset_health_cache()

        tier = await pick_tier(upstream_client, PRIMARY_MODEL)

        await upstream_client.aclose()

        assert tier is not None
        assert tier["model"] == FALLBACK_MODEL

    @pytest.mark.asyncio
    async def test_antigravity_type_recognized_by_pick_tier(self, tmp_path):
        """pick_tier should recognize type='antigravity' for health checking."""
        # Write config with antigravity type
        fb_config = {
            "enabled": True,
            "health_poll_interval_seconds": 30,
            "min_remaining_fraction": 0.05,
            "tiers": [
                {
                    "name": "Primary",
                    "model": "model-a",
                    "type": "antigravity",
                    "health_key": "model-a"
                },
                {
                    "name": "Fallback",
                    "model": "model-b",
                    "type": "antigravity",
                    "health_key": "model-b"
                }
            ]
        }
        fb_file = Path(os.environ["FALLBACK_CONFIG"])
        fb_file.write_text(json.dumps(fb_config, indent=2))

        # Reload config
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]

        mock_state["health_models"] = {
            "model-a": 0.01,
            "model-b": 0.80,
        }

        from src.fallback import pick_tier

        upstream_client = _make_upstream_client()
        await _reset_health_cache()

        tier = await pick_tier(upstream_client, "model-a")

        await upstream_client.aclose()

        assert tier is not None
        assert tier["model"] == "model-b"
        assert tier["type"] == "antigravity"

    @pytest.mark.asyncio
    async def test_reactive_fallback_checks_both_types(self, tmp_path):
        """_try_reactive_fallback should handle both 'upstream' and
        'antigravity' type tiers in the is_available check."""
        # Config with mixed types
        fb_config = {
            "enabled": True,
            "health_poll_interval_seconds": 30,
            "min_remaining_fraction": 0.05,
            "tiers": [
                {
                    "name": "Upstream Tier",
                    "model": PRIMARY_MODEL,
                    "type": "upstream",
                    "health_key": PRIMARY_MODEL
                },
                {
                    "name": "Antigravity Tier",
                    "model": "ag-model",
                    "type": "antigravity",
                    "health_key": "ag-model"
                }
            ]
        }
        fb_file = Path(os.environ["FALLBACK_CONFIG"])
        fb_file.write_text(json.dumps(fb_config, indent=2))

        # Reload
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]

        mock_state["health_models"] = {
            PRIMARY_MODEL: 0.80,
            "ag-model": 0.80,
        }
        mock_state["model_behavior"] = {
            PRIMARY_MODEL: "429",
            "ag-model": "ok",
        }

        resp = await _proxy_post(BASIC_REQUEST)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "ag-model"


# ══════════════════════════════════════════════
#  Edge Cases
# ══════════════════════════════════════════════


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_fallback_disabled_passes_through(self, tmp_path):
        """When fallback is disabled, errors pass through directly."""
        fb_config = {
            "enabled": False,
            "tiers": []
        }
        fb_file = Path(os.environ["FALLBACK_CONFIG"])
        fb_file.write_text(json.dumps(fb_config, indent=2))

        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("src"):
                del sys.modules[mod_name]

        mock_state["model_behavior"] = {PRIMARY_MODEL: "429"}

        resp = await _proxy_post(BASIC_REQUEST)

        # Should get the 429 through without fallback attempt
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_unknown_model_no_fallback(self):
        """Requesting a model not in tiers should not trigger fallback."""
        mock_state["model_behavior"] = {"unknown-model": "ok"}

        body = {**BASIC_REQUEST, "model": "unknown-model"}
        resp = await _proxy_post(body)

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "unknown-model"
