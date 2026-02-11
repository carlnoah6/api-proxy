"""Mock upstream server for fallback testing.

A lightweight FastAPI app simulating Antigravity upstream behavior.
Behavior is controlled via the module-level `mock_state` dict which
tests can modify before making requests.

Supported behaviors per model (set in mock_state["model_behavior"]):
  "ok"             -> 200 with standard Anthropic messages response
  "429"            -> 429 rate limit error (triggers reactive fallback)
  "400_exhausted"  -> 400 with "exhausted" keyword (triggers reactive fallback)
  "503"            -> 503 service unavailable (triggers reactive fallback)
"""
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ── Controllable state ──

mock_state = {
    # model_name -> behavior ("ok", "429", "400_exhausted", "503")
    "model_behavior": {},

    # health_key -> remainingFraction (0.0 - 1.0)
    "health_models": {},
}


def reset_mock_state():
    """Reset mock state to defaults (call between tests)"""
    mock_state["model_behavior"] = {}
    mock_state["health_models"] = {}


# ── Mock ASGI app ──

mock_app = FastAPI(title="Mock Upstream")


@mock_app.get("/health")
async def health():
    """Return health data based on mock_state["health_models"]"""
    models = {}
    for model_key, fraction in mock_state["health_models"].items():
        models[model_key] = {"remainingFraction": fraction}
    return {"accounts": [{"models": models}]}


@mock_app.post("/v1/messages")
async def messages(request: Request):
    """Handle messages endpoint with configurable behavior per model"""
    body = await request.json()
    model = body.get("model", "")
    is_stream = body.get("stream", False)

    behavior = mock_state["model_behavior"].get(model, "ok")

    if behavior == "429":
        return JSONResponse(
            status_code=429,
            content={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": "Rate limit exhausted for this model"
                }
            }
        )

    if behavior == "400_exhausted":
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is exhausted"
                }
            }
        )

    if behavior == "503":
        return JSONResponse(
            status_code=503,
            content={
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Service temporarily unavailable"
                }
            }
        )

    # Normal 200 response
    if is_stream:
        return _stream_response(model)
    else:
        return _non_stream_response(model)


def _non_stream_response(model: str) -> JSONResponse:
    """Standard Anthropic messages API 200 response (non-streaming)"""
    return JSONResponse(content={
        "id": "msg_mock_001",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": f"Hello from {model}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })


def _stream_response(model: str) -> StreamingResponse:
    """Standard Anthropic messages API 200 response (streaming SSE)"""
    async def generate():
        # message_start
        yield _sse("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_mock_stream_001",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        })
        # content_block_start
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })
        # content_block_delta
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"Hello from {model}"}
        })
        # content_block_stop
        yield _sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0
        })
        # message_delta
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5}
        })
        # message_stop
        yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        generate(),
        status_code=200,
        media_type="text/event-stream",
        headers={"content-type": "text/event-stream"}
    )


def _sse(event: str, data: dict) -> str:
    """Format a server-sent event"""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
