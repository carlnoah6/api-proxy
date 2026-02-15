"""FastAPI application - multi-model gateway with usage tracking"""
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import admin as admin_handlers
from . import health as health_module
from .auth import check_model_access, get_accessible_models, require_api_key
from .config import get_models_registry, log
from .usage import record_usage

# ── Model registry (loaded once at startup, refreshable) ──

_models_registry: dict[str, dict] = {}


def _get_model_config(model_id: str) -> dict | None:
    """Look up a model in the registry."""
    global _models_registry
    if not _models_registry:
        _models_registry = get_models_registry()
    return _models_registry.get(model_id)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models_registry
    _models_registry = get_models_registry()
    log.info(f"Loaded {len(_models_registry)} models: {list(_models_registry.keys())}")

    # Create a shared httpx client pool (per-model clients are created on-demand)
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)
    )
    yield
    await app.state.http_client.aclose()


# ── App ──

app = FastAPI(title="Luna API Proxy", lifespan=lifespan)


# ── Generic proxy function ──


async def _proxy_to_upstream(
    model_config: dict,
    request: Request,
    body: bytes,
    req_data: dict | None,
    key_info: dict,
    is_stream: bool,
    client_format: str,
):
    """Proxy a request to the correct upstream based on model_config."""
    http_client: httpx.AsyncClient = request.app.state.http_client

    base_url = model_config["base_url"]
    chat_endpoint = model_config["chat_endpoint"]
    upstream_url = f"{base_url}{chat_endpoint}"

    # Build upstream headers
    headers = {"content-type": "application/json"}
    auth_header = model_config.get("auth_header", "authorization_bearer")
    api_key = model_config.get("api_key", "")

    if auth_header == "x-api-key":
        headers["x-api-key"] = api_key
    else:
        headers["authorization"] = f"Bearer {api_key}"

    # For streaming requests, add stream_options to include usage in final chunk
    # This is supported by OpenAI-compatible APIs (including Moonshot/Kimi)
    if is_stream and req_data:
        req_data = req_data.copy()
        if "stream_options" not in req_data:
            req_data["stream_options"] = {"include_usage": True}
        body = json.dumps(req_data).encode("utf-8")

    if is_stream:
        return await _handle_stream(http_client, upstream_url, headers, body, req_data, key_info, client_format)
    else:
        return await _handle_non_stream(http_client, upstream_url, headers, body, req_data, key_info, client_format)


# ── Stream handler ──


async def _handle_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: bytes,
    req_data: dict | None,
    key_info: dict,
    client_format: str,
):
    upstream_req = client.build_request("POST", url, headers=headers, content=body)
    upstream_resp = await client.send(upstream_req, stream=True)

    if upstream_resp.status_code != 200:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        try:
            error_json = json.loads(error_body)
        except Exception:
            error_json = {"error": error_body.decode("utf-8", errors="ignore")}
        return JSONResponse(content=error_json, status_code=upstream_resp.status_code)

    # Usage tracking wrapper
    input_tokens = 0
    output_tokens = 0
    model = ""

    async def stream_and_count():
        nonlocal input_tokens, output_tokens, model
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
                line_str = chunk.decode("utf-8", errors="ignore")
                for line in line_str.split("\n"):
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        d = json.loads(line[6:])

                        if client_format == "anthropic":
                            msg = d.get("message", {})
                            if isinstance(msg, dict) and "usage" in msg:
                                u = msg["usage"]
                                input_tokens += u.get("input_tokens", 0) or 0
                                output_tokens += u.get("output_tokens", 0) or 0
                            if "usage" in d and "message" not in d:
                                u = d["usage"]
                                input_tokens += u.get("input_tokens", 0) or 0
                                output_tokens += u.get("output_tokens", 0) or 0
                            if isinstance(msg, dict) and "model" in msg:
                                model = msg["model"]
                        else:
                            if "usage" in d:
                                u = d["usage"]
                                input_tokens += u.get("prompt_tokens", 0) or 0
                                output_tokens += u.get("completion_tokens", 0) or 0
                            if "model" in d:
                                model = d["model"]
                    except Exception:
                        pass
            await upstream_resp.aclose()
        except Exception as e:
            log.warning(f"Stream error for model={model or 'unknown'}: {e}")
            try:
                await upstream_resp.aclose()
            except Exception:
                pass

        if input_tokens or output_tokens:
            record_usage(key_info["key"], input_tokens, output_tokens, model)

    return StreamingResponse(
        stream_and_count(),
        status_code=upstream_resp.status_code,
        headers=dict(upstream_resp.headers),
        media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
    )


# ── Non-stream handler ──


async def _handle_non_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: bytes,
    req_data: dict | None,
    key_info: dict,
    client_format: str,
):
    resp = await client.request("POST", url, headers=headers, content=body)

    # Extract usage
    try:
        resp_data = resp.json()
        usage = resp_data.get("usage", {})

        if client_format == "anthropic":
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
        else:
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0

        model = resp_data.get("model", "")
        if input_tokens or output_tokens:
            record_usage(key_info["key"], input_tokens, output_tokens, model)
    except Exception:
        pass

    if resp.headers.get("content-type", "").startswith("application/json"):
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    else:
        return JSONResponse(content={"error": resp.text}, status_code=resp.status_code)


# ── Route Handlers ──


@app.post("/v1/messages")
async def post_messages(request: Request, key_info: dict = Depends(require_api_key)):
    """Anthropic Messages API — route to anthropic-format models."""
    body = await request.body()
    req_data = None
    is_stream = False

    if body:
        try:
            req_data = json.loads(body)
            is_stream = req_data.get("stream", False)
        except Exception:
            pass

    model_id = req_data.get("model", "") if req_data else ""
    log.info(f"[anthropic] Request for model: {model_id}")

    model_config = _get_model_config(model_id)
    if not model_config:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Model '{model_id}' is not available. Use GET /v1/models to list available models.",
                },
            },
        )

    if not check_model_access(key_info, model_id):
        return JSONResponse(
            status_code=403,
            content={
                "type": "error",
                "error": {
                    "type": "permission_error",
                    "message": f"Your API key does not have access to model '{model_id}'.",
                },
            },
        )

    if model_config["format"] != "anthropic":
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Model '{model_id}' uses {model_config['format']} format. Use /v1/chat/completions instead.",
                },
            },
        )

    return await _proxy_to_upstream(model_config, request, body, req_data, key_info, is_stream, "anthropic")


@app.post("/v1/chat/completions")
async def post_chat_completions(request: Request, key_info: dict = Depends(require_api_key)):
    """OpenAI Chat Completions API — route to openai-format models."""
    body = await request.body()
    req_data = None
    is_stream = False

    if body:
        try:
            req_data = json.loads(body)
            is_stream = req_data.get("stream", False)
        except Exception:
            pass

    model_id = req_data.get("model", "") if req_data else ""
    log.info(f"[openai] Request for model: {model_id}")

    model_config = _get_model_config(model_id)
    if not model_config:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model_id}' is not available. Use GET /v1/models to list available models.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    if not check_model_access(key_info, model_id):
        return JSONResponse(
            status_code=403,
            content={
                "error": {
                    "message": f"Your API key does not have access to model '{model_id}'.",
                    "type": "permission_error",
                    "code": "model_access_denied",
                }
            },
        )

    if model_config["format"] != "openai":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model_id}' uses {model_config['format']} format. Use /v1/messages instead.",
                    "type": "invalid_request_error",
                    "code": "wrong_endpoint",
                }
            },
        )

    return await _proxy_to_upstream(model_config, request, body, req_data, key_info, is_stream, "openai")


@app.get("/v1/models")
async def list_models(key_info: dict = Depends(require_api_key)):
    """List all available models from the registry."""
    global _models_registry
    if not _models_registry:
        _models_registry = get_models_registry()

    created_time = 1739347200  # 2025-02-12T00:00:00Z

    accessible = get_accessible_models(key_info, _models_registry)
    models = []
    for model_id, config in accessible.items():
        models.append({
            "id": model_id,
            "object": "model",
            "created": created_time,
            "owned_by": config.get("owned_by", "system"),
            "format": config["format"],
        })

    return JSONResponse(content={"object": "list", "data": models})


# ── Admin routes ──

app.post("/admin/keys")(admin_handlers.admin_create_key)
app.get("/admin/keys")(admin_handlers.admin_list_keys)
app.get("/admin/keys/{api_key}/usage")(admin_handlers.admin_key_usage)
app.post("/admin/keys/{api_key}/disable")(admin_handlers.admin_disable_key)
app.post("/admin/keys/{api_key}/enable")(admin_handlers.admin_enable_key)
app.delete("/admin/keys/{api_key}")(admin_handlers.admin_delete_key)
app.get("/admin/usage")(admin_handlers.admin_total_usage)
app.get("/admin/usage/daily")(admin_handlers.admin_daily_usage)
app.get("/admin/usage/hourly")(admin_handlers.admin_hourly_usage)
app.get("/admin/models")(admin_handlers.admin_models_status)
app.get("/health")(health_module.health)
