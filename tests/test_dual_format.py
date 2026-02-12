
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock, ANY
from fastapi.testclient import TestClient
import httpx

# Mock environment before importing app
import os
os.environ["KEYS_FILE"] = "keys.json"
os.environ["FALLBACK_CONFIG"] = "fallback.json"

from src.app import app, require_api_key

# Mock data
MOCK_KEYS = {
    "keys": {
        "sk-test": {
            "name": "Test User",
            "enabled": 1,
            "usage": {}
        }
    }
}

MOCK_FALLBACK_TIER = {
    "name": "External Tier",
    "model": "external-model",
    "type": "external",
    "base_url": "https://external.api/v1",
    "api_key": "sk-ext"
}

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture(autouse=True)
def mock_auth():
    # Override FastAPI dependency instead of patching
    app.dependency_overrides[require_api_key] = lambda: {"key": "sk-test", "name": "Test User"}
    yield
    app.dependency_overrides = {}

@pytest.fixture
def mock_pick_tier():
    with patch("src.app.pick_tier", new_callable=AsyncMock) as mock:
        mock.return_value = MOCK_FALLBACK_TIER
        yield mock

class TestDualFormat:

    @patch("httpx.AsyncClient")
    def test_anthropic_route_fallback_conversion(self, mock_client_cls, client, mock_pick_tier):
        """Test /v1/messages converts Anthropic -> OpenAI -> Anthropic"""
        
        # Setup External Tier Client Mock
        mock_ext_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_ext_client
        
        # Mock external response (OpenAI format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"role": "assistant", "content": "OpenAI Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_response.headers = {"content-type": "application/json"}
        mock_ext_client.post.return_value = mock_response

        # Request (Anthropic Format)
        anthropic_body = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        response = client.post("/v1/messages", json=anthropic_body)

        # 1. Verify Fallback Triggered
        mock_pick_tier.assert_called_with(ANY, "claude-3")

        # 2. Verify External Call was converted to OpenAI format
        call_args = mock_ext_client.post.call_args
        assert call_args is not None
        _, kwargs = call_args
        sent_body = kwargs["json"]
        
        # Should be converted to OpenAI format
        assert "messages" in sent_body
        assert isinstance(sent_body["messages"], list)
        assert sent_body["messages"][0]["role"] == "user" 
        assert sent_body["messages"][0]["content"] == "Hello"
        assert sent_body["model"] == "external-model" # Model swapped

        # 3. Verify Response converted back to Anthropic
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["type"] == "message"
        assert resp_data["content"][0]["text"] == "OpenAI Response"
        assert "usage" in resp_data

    @patch("httpx.AsyncClient")
    def test_openai_route_fallback_passthrough(self, mock_client_cls, client, mock_pick_tier):
        """Test /v1/chat/completions passes through OpenAI -> OpenAI -> OpenAI"""
        
        # Setup External Tier Client Mock
        mock_ext_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_ext_client
        
        # Mock external response (OpenAI format)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-456",
            "choices": [{"message": {"role": "assistant", "content": "Passthrough Response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_response.headers = {"content-type": "application/json"}
        mock_ext_client.post.return_value = mock_response

        # Request (OpenAI Format)
        openai_body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        response = client.post("/v1/chat/completions", json=openai_body)

        # 1. Verify Fallback Triggered
        mock_pick_tier.assert_called_with(ANY, "gpt-4")

        # 2. Verify External Call passed through (no conversion)
        call_args = mock_ext_client.post.call_args
        _, kwargs = call_args
        sent_body = kwargs["json"]
        
        # Should match input exactly (except model)
        assert sent_body["messages"] == openai_body["messages"]
        assert sent_body["model"] == "external-model"

        # 3. Verify Response passed through
        assert response.status_code == 200
        resp_data = response.json()
        assert "choices" in resp_data # OpenAI format
        assert resp_data["choices"][0]["message"]["content"] == "Passthrough Response"

