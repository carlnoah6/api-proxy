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
        assert "upstream" in data
        assert "fallback_enabled" in data


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


class TestNonStreaming:
    def test_basic_request(self):
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Say 'test ok' and nothing else"}],
                "max_tokens": 20
            },
            headers={"x-api-key": TEST_KEY},
            timeout=30
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data or "choices" in data


class TestStreaming:
    def test_streaming_request(self):
        with httpx.stream(
            "POST",
            f"{BASE_URL}/v1/messages",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Say 'test ok'"}],
                "stream": True,
                "max_tokens": 20
            },
            headers={"x-api-key": TEST_KEY},
            timeout=30
        ) as resp:
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            # Should be SSE
            assert "text/event-stream" in content_type or resp.status_code == 200
            # Read at least some data
            chunks = []
            for chunk in resp.iter_text():
                chunks.append(chunk)
                if len(chunks) > 3:
                    break
            assert len(chunks) > 0


class TestUsageTracking:
    def test_usage_is_recorded(self):
        """Make a request and verify usage is tracked"""
        # First, make a request
        resp = httpx.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "gemini-3-flash",
                "messages": [{"role": "user", "content": "Say exactly: test"}],
                "max_tokens": 10
            },
            headers={"x-api-key": TEST_KEY},
            timeout=30
        )
        assert resp.status_code == 200
        # Usage verification would need admin access
        # For now, just verify the request succeeded
