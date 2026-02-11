"""FastAPI application - routes and proxy logic"""
import json
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import admin as admin_handlers
from . import health as health_module
from . import webhook as webhook_module
from .auth import require_api_key
from .config import UPSTREAM, log
from .fallback import health_cache, load_fallback_config, pick_tier
from .proxy import anthropic_to_openai, openai_stream_to_anthropic_stream, openai_to_anthropic
from .usage import record_usage

# ── External tier call ──


async def call_external_tier(
    tier: dict, body: dict, is_stream: bool,
    key_info: dict, original_model: str
) -> StreamingResponse | JSONResponse:
    """Call external API (e.g., Kimi)"""
    openai_body = anthropic_to_openai(body)
    openai_body["model"] = tier["model"]

    max_limit = tier.get("max_tokens_limit", 8192)
    if openai_body.get("max_tokens", 0) > max_limit:
        openai_body["max_tokens"] = max_limit

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {tier['api_key']}"
    }

    async with httpx.AsyncClient(timeout=300) as ext_client:
        if is_stream:
            openai_body["stream"] = True
            req = ext_client.build_request(
                "POST", f"{tier['base_url']}/chat/completions",
                json=openai_body, headers=headers
            )
            resp = await ext_client.send(req, stream=True)

            if resp.status_code != 200:
                error_body = await resp.aread()
                await resp.aclose()
                return JSONResponse(
                    content={"type": "error", "error": {"type": "api_error", "message": error_body.decode()}},
                    status_code=resp.status_code
                )

            converter = openai_stream_to_anthropic_stream(original_model)

            async def stream_gen():
                async for chunk in converter(resp):
                    yield chunk

            return StreamingResponse(
                stream_gen(),
                status_code=200,
                media_type="text/event-stream",
                headers={"content-type": "text/event-stream"}
            )
        else:
            resp = await ext_client.post(
                f"{tier['base_url']}/chat/completions",
                json=openai_body, headers=headers
            )

            if resp.status_code != 200:
                return JSONResponse(
                    content={"type": "error", "error": {"type": "api_error", "message": resp.text}},
                    status_code=resp.status_code
                )

            resp_data = resp.json()
            anthropic_resp = openai_to_anthropic(resp_data, original_model)

            usage = anthropic_resp.get("usage", {})
            record_usage(
                key_info["key"],
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                tier["model"]
            )

            return JSONResponse(content=anthropic_resp)


# ── Reactive fallback helper ──


async def _try_reactive_fallback(
    client, request, url, headers, req_data, key_info,
    is_stream, error_status, error_body=b""
):
    """Try to find and use a fallback when upstream fails"""
    health_cache.last_poll = 0
    await health_cache.poll(client)

    config = load_fallback_config()
    tiers = config.get("tiers", [])
    requested_model = req_data.get("model", "")

    for tier in tiers:
        if tier["model"] == requested_model:
            continue

        is_avail = False
        if tier["type"] in ("antigravity", "upstream"):
            is_avail = health_cache.is_available(tier.get("health_key", tier["model"]))
        else:
            is_avail = True

        if is_avail:
            log.info(f"⚡ Reactive fallback ({error_status}): {requested_model} → {tier['model']}")
            if tier["type"] in ("antigravity", "upstream"):
                req_data["model"] = tier["model"]
                new_body = json.dumps(req_data).encode()

                if is_stream:
                    upstream_req = client.build_request(
                        method=request.method, url=url, headers=headers, content=new_body
                    )
                    upstream_resp = await client.send(upstream_req, stream=True)
                    if upstream_resp.status_code == 200:
                        return upstream_resp
                    else:
                        await upstream_resp.aclose()
                else:
                    resp = await client.request(
                        method=request.method, url=url, headers=headers, content=new_body
                    )
                    if resp.status_code == 200:
                        return resp
            else:
                return await call_external_tier(
                    tier, req_data, is_stream, key_info, requested_model
                )

    return None  # No fallback succeeded


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        base_url=UPSTREAM,
        timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30)
    )
    try:
        await health_cache.poll(app.state.client)
        log.info(f"Initial health poll: {len(health_cache.models)} models")
    except Exception:
        pass
    yield
    await app.state.client.aclose()


# ── App ──

app = FastAPI(title="Luna API Proxy", lifespan=lifespan)


# ── Proxy route ──


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str, key_info: dict = Depends(require_api_key)):
    client: httpx.AsyncClient = request.app.state.client
    body = await request.body()

    # Build upstream headers (strip auth headers)
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "authorization", "x-api-key", "content-length"):
            headers[k] = v
    headers["x-api-key"] = "test"

    # Parse request body
    is_stream = False
    req_data = None
    if body:
        try:
            req_data = json.loads(body)
            is_stream = req_data.get("stream", False)
        except Exception:
            pass

    # ── Proactive Fallback ──
    if req_data and path == "messages":
        requested_model = req_data.get("model", "")
        log.info(f"Incoming request for model: {requested_model}")
        fallback_tier = await pick_tier(client, requested_model)

        if fallback_tier:
            if fallback_tier["type"] not in ("antigravity", "upstream"):
                return await call_external_tier(
                    fallback_tier, req_data, is_stream, key_info, requested_model)
            else:
                req_data["model"] = fallback_tier["model"]
                body = json.dumps(req_data).encode()
                log.info(f"Swapped model in request: {requested_model} → {fallback_tier['model']}")

    url = f"/v1/{path}"

    if is_stream:
        return await _handle_stream(client, request, url, headers, body, req_data, path, key_info)
    else:
        return await _handle_non_stream(client, request, url, headers, body, req_data, path, key_info)


# ── Stream handler ──


async def _handle_stream(client, request, url, headers, body, req_data, path, key_info):
    upstream_req = client.build_request(
        method=request.method, url=url, headers=headers, content=body
    )
    upstream_resp = await client.send(upstream_req, stream=True)

    # Check for upstream error — reactive fallback
    should_fallback = False
    error_body = b""
    error_status = upstream_resp.status_code

    if req_data and path == "messages":
        if upstream_resp.status_code in (429, 503):
            should_fallback = True
        elif upstream_resp.status_code != 200:
            try:
                error_body = await upstream_resp.aread()
                error_text = error_body.decode("utf-8", errors="ignore").lower()
                if any(x in error_text for x in ("exhausted", "capacity", "quota", "credit", "balance", "rate limit")):
                    should_fallback = True
            except Exception:
                pass

    if should_fallback:
        await upstream_resp.aclose()
        fallback_result = await _try_reactive_fallback(
            client, request, url, headers, req_data, key_info,
            True, error_status, error_body
        )
        if fallback_result is not None:
            if isinstance(fallback_result, (StreamingResponse, JSONResponse)):
                return fallback_result
            upstream_resp = fallback_result
        else:
            return JSONResponse(
                content=json.loads(error_body) if error_body else {"error": "Service Unavailable"},
                status_code=error_status
            )

    # If we read the body but didn't fallback, return that error body
    if not should_fallback and error_body:
        return JSONResponse(
            content=json.loads(error_body) if error_body else {},
            status_code=error_status,
            headers=dict(upstream_resp.headers)
        )

    input_tokens = 0
    output_tokens = 0
    model = ""

    async def stream_and_count():
        nonlocal input_tokens, output_tokens, model
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
                for line in chunk.decode("utf-8", errors="ignore").split("\n"):
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            d = json.loads(line[6:])
                            msg = d.get("message", {})
                            if isinstance(msg, dict) and "usage" in msg:
                                u = msg["usage"]
                                input_tokens += u.get("input_tokens", 0) or 0
                                output_tokens += u.get("output_tokens", 0) or 0
                            if "usage" in d and "message" not in d:
                                u = d["usage"]
                                input_tokens += u.get("input_tokens", 0) or 0
                                output_tokens += u.get("output_tokens", 0) or 0
                            if "model" in d:
                                model = d["model"]
                            elif isinstance(msg, dict) and "model" in msg:
                                model = msg["model"]
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
        media_type=upstream_resp.headers.get("content-type", "text/event-stream")
    )


# ── Non-stream handler ──


async def _handle_non_stream(client, request, url, headers, body, req_data, path, key_info):
    resp = await client.request(
        method=request.method, url=url, headers=headers, content=body
    )

    # Reactive fallback on 429/503/exhausted
    should_fallback = False
    if req_data and path == "messages":
        if resp.status_code in (429, 503):
            should_fallback = True
        elif resp.status_code != 200:
            try:
                error_text = resp.text.lower()
                if any(x in error_text for x in ("exhausted", "capacity", "quota", "credit", "balance", "rate limit")):
                    should_fallback = True
            except Exception:
                pass

    if should_fallback:
        fallback_result = await _try_reactive_fallback(
            client, request, url, headers, req_data, key_info,
            False, resp.status_code
        )
        if fallback_result is not None:
            if isinstance(fallback_result, (StreamingResponse, JSONResponse)):
                return fallback_result
            resp = fallback_result

    # Extract usage
    try:
        resp_data = resp.json()
        usage = resp_data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        model = resp_data.get("model", "")
        if input_tokens or output_tokens:
            record_usage(key_info["key"], input_tokens, output_tokens, model)
    except Exception:
        pass

    return JSONResponse(
        content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text},
        status_code=resp.status_code
    )


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
app.get("/admin/fallback")(admin_handlers.admin_fallback_status)
app.post("/oauth/callback")(admin_handlers.oauth_callback_post)
app.get("/oauth/callback")(admin_handlers.oauth_callback_get)
app.get("/health")(health_module.health)

# ── Webhook routes (no auth required) ──

app.include_router(webhook_module.router)
