"""Sanitize tool history for Aiberm Claude models.

Aiberm cannot handle tool_use/tool_result in message history for Claude models.
Both OpenAI format (/v1/chat/completions) and Anthropic format (/v1/messages)
return 400 "Input is too long" when tool_calls appear in conversation history.

First tool call works fine (tools definition + user message). Only historical
tool interactions trigger the bug. GPT models are unaffected.

This module converts tool interactions in message history into plain text
summaries, preserving the tools definition so new tool calls still work.
"""
import logging

log = logging.getLogger("api-proxy")


def needs_tool_sanitization(provider_id: str, model: str, req_data: dict) -> bool:
    """Check if request needs tool history sanitization."""
    if provider_id != "aiberm":
        return False
    if not model.lower().startswith("claude"):
        return False
    for msg in req_data.get("messages", []):
        if msg.get("role") == "tool" or msg.get("tool_calls"):
            return True
    return False


def sanitize_tool_history(req_data: dict) -> dict:
    """Convert tool_calls/tool messages into plain text summaries.

    Preserves the tools definition so the model can still make new tool calls.
    Only historical tool interactions are converted to text.
    """
    result = req_data.copy()
    messages = req_data.get("messages", [])
    sanitized = []

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role in ("system", "user"):
            sanitized.append(msg.copy())
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            text_parts = []

            # Preserve any text content from the assistant message
            content = msg.get("content")
            if content:
                if isinstance(content, str) and content.strip():
                    text_parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text" and c.get("text", "").strip():
                            text_parts.append(c["text"])

            # Summarize each tool call and its result
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                raw_args = func.get("arguments", "{}")
                args = raw_args if isinstance(raw_args, str) else str(raw_args)
                tc_id = tc.get("id", "")

                # Find matching tool result in subsequent messages
                tool_result = ""
                for j in range(i + 1, min(i + len(tool_calls) + 2, len(messages))):
                    if messages[j].get("role") == "tool" and messages[j].get("tool_call_id") == tc_id:
                        tr_content = messages[j].get("content", "")
                        if isinstance(tr_content, list):
                            tr_content = "\n".join(
                                c.get("text", "") for c in tr_content if isinstance(c, dict)
                            )
                        tool_result = str(tr_content)[:500]
                        break

                text_parts.append(f"[Used tool: {name}({args[:200]})]")
                if tool_result:
                    text_parts.append(f"[Result: {tool_result}]")

            sanitized.append({
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else "[Used tools]"
            })

            # Skip the subsequent tool result messages
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                i += 1
            continue

        if role == "assistant":
            sanitized.append(msg.copy())
            i += 1
            continue

        if role == "tool":
            # Orphan tool result (no preceding assistant with tool_calls)
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            sanitized.append({
                "role": "user",
                "content": f"[Previous tool result: {str(content)[:500]}]"
            })
            i += 1
            continue

        # Unknown role, pass through
        sanitized.append(msg)
        i += 1

    # Merge consecutive same-role messages (required by some APIs)
    merged = []
    for msg in sanitized:
        if merged and merged[-1].get("role") == msg.get("role"):
            prev = merged[-1].get("content", "")
            curr = msg.get("content", "")
            if isinstance(prev, str) and isinstance(curr, str):
                merged[-1]["content"] = prev + "\n" + curr
            else:
                merged.append(msg)
        else:
            merged.append(msg)

    result["messages"] = merged
    return result
