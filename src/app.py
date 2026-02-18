"""FastAPI application - multi-model gateway with provider-based routing and ACL"""
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import admin as admin_handlers
from . import health as health_module
from .auth import get_accessible_providers, require_api_key
from .config import get_known_models, get_providers, get_routing, log, resolve_model
from .usage import record_usage

# ── Providers (loaded at startup, refreshed from config as fallback) ──

_providers: dict[str, dict] = {}


def _get_providers() -> dict[str, dict]:
    """Return providers, loading from config if not yet initialized."""
    global _providers
    if not _providers:
        _providers = get_providers()  # lifespan init
    return _providers


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _providers
    _providers = get_providers()  # lifespan init
    log.info(f"Loaded {len(_providers)} providers: {list(_providers.keys())}")

    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)
    )
    yield
    await app.state.http_client.aclose()


# ── App ──

app = FastAPI(title="Luna API Proxy", lifespan=lifespan)


# ── Generic proxy function ──


async def _proxy_to_upstream(
    provider: dict,
    request: Request,
    body: bytes,
    req_data: dict | None,
    key_info: dict,
    is_stream: bool,
    client_format: str,
):
    """Proxy a request to the correct upstream based on provider config."""
    http_client: httpx.AsyncClient = request.app.state.http_client

    # Store provider info for potential 400-retry-with-sanitize
    _pid = provider.get("_id", "")
    _model = req_data.get("model", "") if req_data else ""

    upstream_url = f"{provider['base_url']}{provider['chat_endpoint']}"

    # Build upstream headers
    headers = {"content-type": "application/json"}
    auth_header = provider.get("auth_header", "authorization_bearer")
    api_key = provider.get("api_key", "")

    if auth_header == "x-api-key":
        headers["x-api-key"] = api_key
    else:
        headers["authorization"] = f"Bearer {api_key}"

    # For streaming, add stream_options for usage tracking
    if is_stream and req_data:
        req_data = req_data.copy()
        if "stream_options" not in req_data:
            req_data["stream_options"] = {"include_usage": True}
        body = json.dumps(req_data).encode("utf-8")

    if is_stream:
        return await _handle_stream(
            http_client, upstream_url, headers, body, req_data, key_info, client_format,
            provider_id=_pid, model_name=_model,
        )
    else:
        return await _handle_non_stream(
            http_client, upstream_url, headers, body, req_data, key_info, client_format,
            provider_id=_pid, model=_model,
        )


# ── Stream handler ──


async def _handle_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: bytes,
    req_data: dict | None,
    key_info: dict,
    client_format: str,
    provider_id: str = "",
    model_name: str = "",
):
    upstream_req = client.build_request("POST", url, headers=headers, content=body)
    upstream_resp = await client.send(upstream_req, stream=True)

    if upstream_resp.status_code != 200:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()

        # Retry on 400 with tool sanitization (Aiberm Claude bug)
        if upstream_resp.status_code == 400 and req_data and provider_id:
            from .tool_sanitizer import needs_tool_sanitization, sanitize_tool_history
            if needs_tool_sanitization(provider_id, model_name, req_data):
                log.info(f"[400-retry] Got 400 (stream) for {model_name} on {provider_id}, retrying with sanitized tool history")
                try:
                    sanitized = sanitize_tool_history(req_data)
                    retry_body = json.dumps(sanitized).encode("utf-8")
                    retry_req = client.build_request("POST", url, headers=headers, content=retry_body)
                    upstream_resp = await client.send(retry_req, stream=True)
                    if upstream_resp.status_code == 200:
                        log.info(f"[400-retry] Stream retry succeeded for {model_name}")
                        # Fall through to normal streaming below
                    else:
                        retry_error = await upstream_resp.aread()
                        await upstream_resp.aclose()
                        try:
                            error_json = json.loads(retry_error)
                        except Exception:
                            error_json = {"error": retry_error.decode("utf-8", errors="ignore")}
                        return JSONResponse(content=error_json, status_code=upstream_resp.status_code)
                except Exception as e:
                    log.warning(f"[400-retry] Stream sanitize failed: {e}")
                    try:
                        error_json = json.loads(error_body)
                    except Exception:
                        error_json = {"error": error_body.decode("utf-8", errors="ignore")}
                    return JSONResponse(content=error_json, status_code=400)
            else:
                try:
                    error_json = json.loads(error_body)
                except Exception:
                    error_json = {"error": error_body.decode("utf-8", errors="ignore")}
                return JSONResponse(content=error_json, status_code=upstream_resp.status_code)
        else:
            try:
                error_json = json.loads(error_body)
            except Exception:
                error_json = {"error": error_body.decode("utf-8", errors="ignore")}
            return JSONResponse(content=error_json, status_code=upstream_resp.status_code)

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
    provider_id: str = "",
    model: str = "",
):
    resp = await client.request("POST", url, headers=headers, content=body)

    # Retry on 400 with tool sanitization (Aiberm Claude bug)
    if resp.status_code == 400 and req_data and provider_id:
        from .tool_sanitizer import needs_tool_sanitization, sanitize_tool_history
        if needs_tool_sanitization(provider_id, model, req_data):
            log.info(f"[400-retry] Got 400 for {model} on {provider_id}, retrying with sanitized tool history")
            try:
                sanitized = sanitize_tool_history(req_data)
                retry_body = json.dumps(sanitized).encode("utf-8")
                resp = await client.request("POST", url, headers=headers, content=retry_body)
                if resp.status_code == 200:
                    log.info(f"[400-retry] Retry succeeded for {model}")
                else:
                    log.warning(f"[400-retry] Retry also failed: {resp.status_code}")
            except Exception as e:
                log.warning(f"[400-retry] Sanitize failed, returning original 400: {e}")

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


# ── Resolve + ACL check helper ──


def _resolve_and_check(model_id: str, key_info: dict, error_format: str = "openai"):
    """Resolve model to provider and check access.

    Passes allowed_providers into resolve_model so the routing chain
    automatically skips providers the user can't access, falling back
    to the next viable provider in the chain.
    """
    allowed = key_info.get("allowed_providers")
    allowed_set = set(allowed) if allowed else None
    provider, provider_id = resolve_model(model_id, _get_providers(), allowed_set)

    if not provider:
        if error_format == "anthropic":
            return None, JSONResponse(
                status_code=400,
                content={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Model '{model_id}' is not available.",
                    },
                },
            )
        return None, JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model_id}' is not available.",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    provider = provider.copy()
    provider["_id"] = provider_id
    return provider, None


# ── Route Handlers ──


@app.post("/v1/messages")
async def post_messages(request: Request, key_info: dict = Depends(require_api_key)):
    """Anthropic Messages API."""
    body = await request.body()
    req_data = json.loads(body) if body else {}
    is_stream = req_data.get("stream", False)
    model_id = req_data.get("model", "")
    log.info(f"[anthropic] Request for model: {model_id}")

    provider, error = _resolve_and_check(model_id, key_info, "anthropic")
    if error:
        return error
    if provider.get("format") != "anthropic":
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": f"Model '{model_id}' uses {provider.get('format')} format. Use /v1/chat/completions.",
                },
            },
        )
    return await _proxy_to_upstream(
        provider, request, body, req_data, key_info, is_stream, "anthropic"
    )


@app.post("/v1/chat/completions")
async def post_chat_completions(
    request: Request, key_info: dict = Depends(require_api_key)
):
    """OpenAI Chat Completions API."""
    body = await request.body()
    req_data = json.loads(body) if body else {}
    is_stream = req_data.get("stream", False)
    model_id = req_data.get("model", "")
    log.info(f"[openai] Request for model: {model_id}")

    provider, error = _resolve_and_check(model_id, key_info, "openai")
    if error:
        return error
    if provider.get("format") != "openai":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model_id}' uses {provider.get('format')} format. Use /v1/messages.",
                    "type": "invalid_request_error",
                    "code": "wrong_endpoint",
                }
            },
        )
    return await _proxy_to_upstream(
        provider, request, body, req_data, key_info, is_stream, "openai"
    )


@app.get("/v1/models")
async def list_models(key_info: dict = Depends(require_api_key)):
    """List available models (filtered by user's provider access)."""
    accessible_ids = set(get_accessible_providers(key_info, _get_providers()).keys())
    routing = get_routing()

    models = []
    seen = set()
    for m in get_known_models():
        mid = m["id"]
        if mid in seen:
            continue
        # Check: is this model reachable via routing chain for this user?
        model_lower = mid.lower()
        reachable = False
        for prefix, chain in routing.items():
            if model_lower.startswith(prefix.lower()):
                for pid in chain:
                    if pid in accessible_ids:
                        reachable = True
                        break
                break
        # Fallback: direct provider check
        if not reachable and m.get("provider") in accessible_ids:
            reachable = True
        if reachable:
            seen.add(mid)
            models.append(
                {
                    "id": mid,
                    "object": "model",
                    "created": 1739347200,
                    "owned_by": m.get("owned_by", "unknown"),
                    "provider": m.get("provider", ""),
                }
            )

    return JSONResponse(content={"object": "list", "data": models})


# ── Admin + Health routes ──

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
