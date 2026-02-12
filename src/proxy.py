"""Format conversion between Anthropic Messages API and OpenAI Chat Completions API"""
import json
import uuid


def anthropic_to_openai(body: dict) -> dict:
    """Anthropic Messages API → OpenAI Chat Completions API"""
    messages = []

    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # Skip thinking blocks in input
                # Skip unsupported blocks in cross-format conversion
                elif block.get("type") in ("tool_use", "tool_result", "image"):
                    pass
            
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})

    result = {
        "model": body.get("model", ""),
        "messages": messages,
    }

    # Map parameters
    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if body.get("stream"):
        result["stream"] = True
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    return result


def openai_to_anthropic(resp_data: dict, original_model: str = "") -> dict:
    """OpenAI Chat Completions response → Anthropic Messages response"""
    content = []

    choices = resp_data.get("choices", [])
    if choices:
        choice = choices[0]
        msg = choice.get("message", {})

        # Handle reasoning_content (Kimi's thinking)
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            content.append({
                "type": "thinking",
                "thinking": reasoning,
                "signature": ""
            })

        # Main text content
        text = msg.get("content", "")
        if text:
            content.append({
                "type": "text",
                "text": text
            })

        # Map stop reason
        finish = choice.get("finish_reason", "")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }.get(finish, "end_turn")
    else:
        content = [{"type": "text", "text": ""}]
        stop_reason = "end_turn"

    # Map usage
    usage = resp_data.get("usage", {})

    return {
        "id": f"msg_{resp_data.get('id', uuid.uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "model": original_model or resp_data.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


def openai_stream_to_anthropic_stream(original_model: str = ""):
    """Generator factory: converts OpenAI SSE stream to Anthropic SSE stream"""

    async def converter(response):
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        sent_start = False
        full_text = ""
        reasoning_text = ""
        input_tokens = 0
        output_tokens = 0
        thinking_done = False
        content_idx = 0

        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            # Send message_start on first chunk
            if not sent_start:
                start_event = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": original_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                }
                yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"
                sent_start = True

            delta = chunk.get("choices", [{}])[0].get("delta", {})

            # Handle reasoning_content (thinking)
            reasoning_delta = delta.get("reasoning_content", "")
            if reasoning_delta:
                if not reasoning_text:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                reasoning_text += reasoning_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_idx, 'delta': {'type': 'thinking_delta', 'thinking': reasoning_delta}})}\n\n"

            # Handle content
            text_delta = delta.get("content", "")
            if text_delta:
                if reasoning_text and not thinking_done:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                    content_idx += 1
                    thinking_done = True
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                elif not full_text and not reasoning_text:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

                full_text += text_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_idx, 'delta': {'type': 'text_delta', 'text': text_delta}})}\n\n"

            # Usage from chunk
            u = chunk.get("usage", {})
            if u.get("prompt_tokens"):
                input_tokens = u["prompt_tokens"]  # noqa: F841
            if u.get("completion_tokens"):
                output_tokens = u["completion_tokens"]

        # Close last content block
        if full_text or reasoning_text:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"

        # message_delta (stop reason)
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

        # message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        await response.aclose()

    return converter
