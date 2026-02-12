"""Integration tests for API Proxy (requires running service)"""
import os

import httpx
import pytest

# Integration tests run against a live server
# Set API_PROXY_TEST_PORT env var to change port (default: 8181 for dev)
TEST_PORT = int(os.environ.get("API_PROXY_TEST_PORT", "8181"))
TEST_KEY = os.environ.get("API_PROXY_TEST_KEY", "sk-dev-test-key")
BASE_URL = f"http://localhost:{TEST_PORT}"


def is_server_running():
    """Check if the server is running"""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_server_running(),
    reason=f"API Proxy not running on port {TEST_PORT}"
)


class TestHealth:
    def test_health_endpoint(self):
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models" in data
        assert "total_models" in data


class TestAuth:
    def test_no_key_returns_401(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            timeout=10
        )
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            headers={"x-api-key": "sk-invalid-key-12345"},
            timeout=10
        )
        assert resp.status_code == 401

    def test_bearer_auth_also_works(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            headers={"Authorization": "Bearer sk-invalid-key-12345"},
            timeout=10
        )
        assert resp.status_code == 401


class TestModelRouting:
    def test_unknown_model_returns_400(self):
        """Requesting an unknown model should return 400, not 500"""
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10
            },
            headers={"x-api-key": TEST_KEY},
            timeout=10
        )
        assert resp.status_code == 400

    def test_models_endpoint(self):
        """GET /v1/models should list available models"""
        resp = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"x-api-key": TEST_KEY},
            timeout=10
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
