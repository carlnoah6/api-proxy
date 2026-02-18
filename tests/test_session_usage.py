"""Tests for per-session usage tracking."""
import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def temp_keys_env():
    data = {
        "admin_key": "sk-admin-test",
        "keys": {
            "sk-test-key": {
                "name": "Test",
                "enabled": True,
                "created": "2026-01-01T00:00:00+08:00",
                "usage": {
                    "total_input": 0,
                    "total_output": 0,
                    "total_requests": 0,
                },
            }
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    p = Path(path)
    with patch("src.auth.KEYS_FILE", p), patch("src.config.KEYS_FILE", p):
        yield path
    os.unlink(path)


class TestSessionUsageTracking:
    def test_record_with_session_id(self):
        from src.usage import record_usage, get_session_usage
        record_usage("sk-test-key", 1000, 500, "claude", session_id="task-abc")
        record_usage("sk-test-key", 2000, 800, "claude", session_id="task-abc")
        usage = get_session_usage("sk-test-key", "task-abc")
        assert usage is not None
        assert usage["input"] == 3000
        assert usage["output"] == 1300
        assert usage["requests"] == 2

    def test_record_without_session_id(self):
        from src.usage import record_usage, get_session_usage
        record_usage("sk-test-key", 1000, 500, "claude")
        usage = get_session_usage("sk-test-key", "")
        assert usage is None

    def test_multiple_sessions(self):
        from src.usage import record_usage, get_session_usage
        record_usage("sk-test-key", 1000, 500, "claude", session_id="t1")
        record_usage("sk-test-key", 2000, 800, "claude", session_id="t2")
        assert get_session_usage("sk-test-key", "t1")["input"] == 1000
        assert get_session_usage("sk-test-key", "t2")["input"] == 2000

    def test_session_not_found(self):
        from src.usage import get_session_usage
        assert get_session_usage("sk-test-key", "nope") is None

    def test_invalid_key(self):
        from src.usage import get_session_usage
        assert get_session_usage("sk-invalid", "t1") is None
