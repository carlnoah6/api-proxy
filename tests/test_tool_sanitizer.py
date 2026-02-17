"""Tests for tool_sanitizer module."""
import pytest
from src.tool_sanitizer import needs_tool_sanitization, sanitize_tool_history


class TestNeedsToolSanitization:
    def test_non_aiberm_provider(self):
        assert not needs_tool_sanitization("openai", "claude-opus-4-6", {
            "messages": [{"role": "tool", "content": "hi"}]
        })

    def test_non_claude_model(self):
        assert not needs_tool_sanitization("aiberm", "gpt-4o", {
            "messages": [{"role": "tool", "content": "hi"}]
        })

    def test_no_tool_history(self):
        assert not needs_tool_sanitization("aiberm", "claude-opus-4-6", {
            "messages": [{"role": "user", "content": "hi"}]
        })

    def test_with_tool_message(self):
        assert needs_tool_sanitization("aiberm", "claude-opus-4-6", {
            "messages": [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]
        })

    def test_with_tool_calls(self):
        assert needs_tool_sanitization("aiberm", "claude-opus-4-6-thinking", {
            "messages": [{"role": "assistant", "tool_calls": [{"id": "tc1"}]}]
        })


class TestSanitizeToolHistory:
    def test_basic_tool_call_and_result(self):
        req = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Run ls"},
                {"role": "assistant", "content": "Let me run that.", "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "exec", "arguments": '{"cmd":"ls"}'}}
                ]},
                {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "tc1"},
                {"role": "user", "content": "thanks"}
            ]
        }
        result = sanitize_tool_history(req)
        msgs = result["messages"]
        # Should have: system, user, assistant(text), user
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "[Used tool: exec" in msgs[2]["content"]
        assert "[Result: file1.txt" in msgs[2]["content"]
        assert "Let me run that." in msgs[2]["content"]
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "thanks"

    def test_orphan_tool_result(self):
        req = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "tool", "content": "orphan result", "tool_call_id": "tc0"},
                {"role": "user", "content": "hi"}
            ]
        }
        result = sanitize_tool_history(req)
        msgs = result["messages"]
        assert msgs[1]["role"] == "user"
        assert "[Previous tool result:" in msgs[1]["content"]

    def test_preserves_normal_messages(self):
        req = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "bye"}
            ]
        }
        result = sanitize_tool_history(req)
        assert result["messages"] == req["messages"]

    def test_multiple_tool_calls(self):
        req = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "do stuff"},
                {"role": "assistant", "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.txt"}'}},
                    {"id": "tc2", "type": "function", "function": {"name": "exec", "arguments": '{"cmd":"ls"}'}}
                ]},
                {"role": "tool", "content": "file content", "tool_call_id": "tc1"},
                {"role": "tool", "content": "dir listing", "tool_call_id": "tc2"},
                {"role": "user", "content": "ok"}
            ]
        }
        result = sanitize_tool_history(req)
        msgs = result["messages"]
        assert len(msgs) == 4
        assert "[Used tool: read" in msgs[2]["content"]
        assert "[Used tool: exec" in msgs[2]["content"]
        assert "[Result: file content]" in msgs[2]["content"]
        assert "[Result: dir listing]" in msgs[2]["content"]

    def test_merges_consecutive_same_role(self):
        req = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "tool", "content": "r1", "tool_call_id": "tc1"},
                {"role": "tool", "content": "r2", "tool_call_id": "tc2"},
                {"role": "user", "content": "hi"}
            ]
        }
        result = sanitize_tool_history(req)
        msgs = result["messages"]
        # Two orphan tools -> two user messages -> merged into one
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 1  # all user messages merged into one
        assert "[Previous tool result: r1]" in user_msgs[0]["content"]
        assert "hi" in user_msgs[0]["content"]

    def test_truncates_long_results(self):
        req = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "go"},
                {"role": "assistant", "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "exec", "arguments": '{}'}}
                ]},
                {"role": "tool", "content": "x" * 1000, "tool_call_id": "tc1"},
                {"role": "user", "content": "ok"}
            ]
        }
        result = sanitize_tool_history(req)
        assistant_msg = [m for m in result["messages"] if m["role"] == "assistant"][0]
        # Result should be truncated to 500 chars
        assert len(assistant_msg["content"]) < 700

    def test_preserves_other_fields(self):
        req = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "claude-opus-4-6",
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "max_tokens": 100
        }
        result = sanitize_tool_history(req)
        assert result["model"] == "claude-opus-4-6"
        assert result["tools"] == req["tools"]
        assert result["max_tokens"] == 100
