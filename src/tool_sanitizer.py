"""Tool history sanitizer — DISABLED.

This module was designed to convert tool_calls/tool_result in message history
into plain text summaries for providers that cannot handle them (e.g. Aiberm
with Claude models).

DISABLED because sanitization causes an infinite loop:
1. Aiberm returns 400 for requests with tool history
2. Sanitizer converts tool history to text summaries like
   "[Previously executed exec — args: ...]"
3. Claude sees these text summaries and does NOT recognize them as
   previously executed tools
4. Claude re-invokes the exact same tools
5. New tool results get added to history → next request has tool history
   → Aiberm 400 again → sanitize again → loop forever

The generic 400 retry (same payload) is kept for transient errors.
If the retry also fails, the 400 propagates to the caller (OpenClaw gateway),
which triggers fallback to the next model in the chain.
"""
import logging

log = logging.getLogger("api-proxy")


def needs_tool_sanitization(provider_id: str, model: str, req_data: dict) -> bool:
    """Always returns False — sanitization is permanently disabled."""
    return False


def sanitize_tool_history(req_data: dict) -> dict:
    """No-op — returns input unchanged. Kept for interface compatibility."""
    return req_data
