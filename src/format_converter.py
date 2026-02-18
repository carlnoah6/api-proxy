"""OpenAI <-> Anthropic format converter for api-proxy.

Converts between OpenAI Chat Completions and Anthropic Messages format
so Claude models can use /v1/chat/completions while talking Anthropic
native format to upstream (which properly supports tool history).
"""
import json
import logging
import re

log = logging.getLogger("api-proxy")


def openai_to_anthropic(req_data: dict) -> dict:
    """Convert OpenAI chat completions request to Anthropic messages format."""
    result = {
        "model": req_data.get("model", ""),
        "max_tokens": req_data.get("max_tokens") or 4096,
    }

    if req_data.get("stream"):
        result["stream"] = True
    if req_data.get("temperature") is not None:
        result["temperature"] = req_data["temperature"]
    if req_data.get("top_p") is not None:
        result["top_p"] = req_data["top_p"]
    if req_data.get("stop"):
        stop = req_data["stop"]
        result["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    system_parts = []
    messages = []

    for msg in req_data.get("messages", []):
        role = msg.get("role", "")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        system_parts.append(c["text"])
            continue

        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append({"type": "text", "text": c["text"]})
                    else:
                        parts.append(c)
                messages.append({"role": "user", "content": parts or content})
            else:
                messages.append({"role": "user", "content": str(content)})

        elif role == "assistant":
            content_parts = []
            text = msg.get("content")
            if text and isinstance(text, str) and text.strip():
                content_parts.append({"type": "text", "text": text})
            elif isinstance(text, list):
                for c in text:
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text", "").strip():
                        content_parts.append(c)

            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": str(args_str)}
                content_parts.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": args,
                })

            if not content_parts:
                content_parts.append({"type": "text", "text": ""})
            messages.append({"role": "assistant", "content": content_parts})

        elif role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if messages and messages[-1]["role"] == "user":
                prev = messages[-1]["content"]
                if isinstance(prev, str):
                    messages[-1]["content"] = [{"type": "text", "text": prev}, tool_result]
                elif isinstance(prev, list):
                    prev.append(tool_result)
            else:
                messages.append({"role": "user", "content": [tool_result]})

    if system_parts:
        result["system"] = "\n\n".join(system_parts)

    # Merge consecutive same-role messages (Anthropic requires alternating)
    merged = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            prev = merged[-1]["content"]
            curr = msg["content"]
            if isinstance(prev, str):
                prev = [{"type": "text", "text": prev}]
            if isinstance(curr, str):
                curr = [{"type": "text", "text": curr}]
            if isinstance(prev, list) and isinstance(curr, list):
                merged[-1]["content"] = prev + curr
        else:
            merged.append(msg)
    result["messages"] = merged

    # Convert tools
    if req_data.get("tools"):
        anthropic_tools = []
        for tool in req_data["tools"]:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        if anthropic_tools:
            result["tools"] = anthropic_tools

    return result


def anthropic_to_openai(resp_data: dict, model: str = "") -> dict:
    """Convert Anthropic messages response to OpenAI chat completions format."""
    text_parts = []
    tool_calls = []

    for block in resp_data.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            # Strip thinking tags
            text = re.sub(r"<thinking>.*?</thinking>\s*", "", text, flags=re.DOTALL).strip()
            if text:
                text_parts.append(text)
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    content = "\n".join(text_parts) if text_parts else None

    # Map stop_reason
    stop_reason = resp_data.get("stop_reason", "end_turn")
    finish_reason = "stop"
    if stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content:
            message["content"] = None

    usage = resp_data.get("usage", {})

    return {
        "id": resp_data.get("id", ""),
        "object": "chat.completion",
        "model": model or resp_data.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def anthropic_stream_to_openai_stream(event: dict, model: str = "") -> dict | None:
    """Convert a parsed Anthropic SSE event dict to OpenAI chunk dict.

    Returns None if the event should be skipped.
    """
    data = event
    event_type = data.get("type", "")

    if event_type == "message_start":
        msg = data.get("message", {})
        usage = msg.get("usage", {})
        chunk = {
            "id": msg.get("id", ""),
            "object": "chat.completion.chunk",
            "model": model or msg.get("model", ""),
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        }
        if usage:
            chunk["usage"] = {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": 0, "total_tokens": usage.get("input_tokens", 0)}
        return chunk

    elif event_type == "content_block_start":
        block = data.get("content_block", {})
        if block.get("type") == "tool_use":
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": data.get("index", 0),
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {"name": block.get("name", ""), "arguments": ""},
                    }]
                }, "finish_reason": None}],
            }
            return chunk
        return None

    elif event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            text = delta.get("text", "")
            # Strip thinking tags from stream
            text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
            if not text:
                return None
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            return chunk
        elif delta.get("type") == "input_json_delta":
            chunk = {
                "id": "",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": data.get("index", 0),
                        "function": {"arguments": delta.get("partial_json", "")},
                    }]
                }, "finish_reason": None}],
            }
            return chunk
        return None

    elif event_type == "message_delta":
        delta = data.get("delta", {})
        stop_reason = delta.get("stop_reason", "end_turn")
        finish_reason = "stop"
        if stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "max_tokens":
            finish_reason = "length"

        usage = data.get("usage", {})
        chunk = {
            "id": "",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        if usage:
            chunk["usage"] = {
                "prompt_tokens": 0,
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("output_tokens", 0),
            }
        return chunk

    elif event_type == "message_stop":
        return None

    return None
