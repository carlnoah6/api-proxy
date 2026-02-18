"""Tests for tool_sanitizer module — sanitization permanently disabled."""
from src.tool_sanitizer import needs_tool_sanitization, sanitize_tool_history


class TestNeedsToolSanitization:
    """All cases should return False — sanitization is disabled."""

    def test_always_false_for_aiberm_claude(self):
        assert not needs_tool_sanitization("aiberm", "claude-opus-4-6", {
            "messages": [{"role": "tool", "content": "result", "tool_call_id": "tc1"}]
        })

    def test_always_false_for_aiberm_claude_thinking(self):
        assert not needs_tool_sanitization("aiberm", "claude-opus-4-6-thinking", {
            "messages": [{"role": "assistant", "tool_calls": [{"id": "tc1"}]}]
        })

    def test_always_false_for_other_providers(self):
        assert not needs_tool_sanitization("openai", "claude-opus-4-6", {
            "messages": [{"role": "tool", "content": "hi"}]
        })

    def test_always_false_for_non_claude(self):
        assert not needs_tool_sanitization("aiberm", "gpt-4o", {
            "messages": [{"role": "tool", "content": "hi"}]
        })


class TestSanitizeToolHistory:
    """sanitize_tool_history is a no-op — returns input unchanged."""

    def test_noop(self):
        req = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "tc1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}
                ]},
                {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            ],
            "model": "claude-opus-4-6"
        }
        result = sanitize_tool_history(req)
        assert result is req  # Same object, no copy
