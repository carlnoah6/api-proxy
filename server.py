#!/usr/bin/env python3
"""
API Key 鉴权代理 — 在 Antigravity (localhost:8080) 前面加一层
功能：API Key 鉴权、用量统计、Key 管理、智能 Fallback
"""
import json, os, time, uuid, hashlib, asyncio, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse

# ── 配置 ──────────────────────────────────────────
UPSTREAM = os.environ.get("UPSTREAM_URL", "http://localhost:8080")
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8180"))
KEYS_FILE = Path(os.environ.get("KEYS_FILE", "/home/ubuntu/api-proxy/keys.json"))
FALLBACK_CONFIG_FILE = Path(os.environ.get("FALLBACK_CONFIG", "/home/ubuntu/api-proxy/fallback.json"))
SGT = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api-proxy")

# ── Fallback 配置 ─────────────────────────────────
def load_fallback_config() -> dict:
    default = {
        "enabled": True,
        "health_poll_interval_seconds": 30,
        "min_remaining_fraction": 0.05,
        "tiers": [
            {
                "name": "Claude Opus 4.6",
                "model": "claude-opus-4-6-thinking",
                "type": "antigravity",
                "health_key": "claude-opus-4-6-thinking"
            },
            {
                "name": "Gemini 3 Pro High",
                "model": "gemini-3-pro-high",
                "type": "antigravity",
                "health_key": "gemini-3-pro-high"
            }
        ]
    }
    if FALLBACK_CONFIG_FILE.exists():
        try:
            return json.loads(FALLBACK_CONFIG_FILE.read_text())
        except:
            pass
    # Write default config
    FALLBACK_CONFIG_FILE.write_text(json.dumps(default, indent=2, ensure_ascii=False))
    return default

# ── Health Cache ──────────────────────────────────
class HealthCache:
    """缓存 Antigravity /health 接口返回的各模型额度"""
    
    def __init__(self):
        self.models: dict = {}  # model -> {remaining: float, resetTime: str}
        self.last_poll: float = 0
        self.poll_interval: int = 30
        self._lock = asyncio.Lock()
    
    async def poll(self, client: httpx.AsyncClient):
        """轮询 /health 更新额度缓存"""
        now = time.time()
        if now - self.last_poll < self.poll_interval:
            return
        
        async with self._lock:
            # Double check after acquiring lock
            if time.time() - self.last_poll < self.poll_interval:
                return
            try:
                resp = await client.get(f"{UPSTREAM}/health", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    accounts = data.get("accounts", [])
                    # Aggregate models from ALL accounts (take max remaining)
                    aggregated_models = {}
                    for acc in accounts:
                        models = acc.get("models", {})
                        for model_name, info in models.items():
                            if model_name not in aggregated_models:
                                aggregated_models[model_name] = info
                            else:
                                # Take the one with higher remainingFraction
                                current_rem = aggregated_models[model_name].get("remainingFraction", 0)
                                new_rem = info.get("remainingFraction", 0)
                                if new_rem > current_rem:
                                    aggregated_models[model_name] = info
                    
                    self.models = aggregated_models
                    self.last_poll = time.time()
                    log.debug(f"Health poll: {len(self.models)} models updated from {len(accounts)} accounts")
            except Exception as e:
                log.warning(f"Health poll failed: {e}")
    
    def get_remaining(self, model_key: str) -> float:
        """获取模型剩余额度比例 (0.0-1.0)，未知则返回 -1"""
        info = self.models.get(model_key, {})
        return info.get("remainingFraction", -1)
    
    def is_available(self, model_key: str, min_remaining: float = 0.05) -> bool:
        """模型是否有足够额度"""
        remaining = self.get_remaining(model_key)
        if remaining < 0:
            return True  # Unknown → assume available
        return remaining > min_remaining


health_cache = HealthCache()

# ── 格式转换：Anthropic ↔ OpenAI ─────────────────

def anthropic_to_openai(body: dict) -> dict:
    """Anthropic Messages API → OpenAI Chat Completions API"""
    messages = []
    
    # System prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be a list of content blocks
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
            # Anthropic content blocks → concatenate text parts
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    pass  # Skip thinking blocks in input
                elif block.get("type") == "tool_use":
                    text_parts.append(f"[Tool call: {block.get('name', '')}({json.dumps(block.get('input', {}))})]")
                elif block.get("type") == "tool_result":
                    text_parts.append(f"[Tool result: {block.get('content', '')}]")
                elif block.get("type") == "image":
                    text_parts.append("[Image]")
            messages.append({"role": role, "content": "\n".join(text_parts) if text_parts else ""})
    
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
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }
        stop_reason = stop_reason_map.get(finish, "end_turn")
    else:
        content = [{"type": "text", "text": ""}]
        stop_reason = "end_turn"
    
    # Map usage
    usage = resp_data.get("usage", {})
    
    return {
        "id": f"msg_{resp_data.get('id', uuid.uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": original_model or resp_data.get("model", ""),
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0
        }
    }


def openai_stream_to_anthropic_stream(original_model: str = ""):
    """Generator factory: converts OpenAI SSE stream to Anthropic SSE stream"""
    
    async def converter(response: httpx.Response):
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        input_tokens = 0
        output_tokens = 0
        full_text = ""
        reasoning_text = ""
        sent_start = False
        in_thinking = False
        thinking_done = False
        content_idx = 0
        
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            
            try:
                chunk = json.loads(data_str)
            except:
                continue
            
            # Send message_start on first chunk
            if not sent_start:
                start_event = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model or chunk.get("model", ""),
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
                if not in_thinking:
                    in_thinking = True
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                reasoning_text += reasoning_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_idx, 'delta': {'type': 'thinking_delta', 'thinking': reasoning_delta}})}\n\n"
            
            # Handle content
            text_delta = delta.get("content", "")
            if text_delta:
                if in_thinking and not thinking_done:
                    # Close thinking block, start text block
                    thinking_done = True
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
                    content_idx += 1
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                elif not in_thinking and not full_text:
                    # First text content, no thinking before it
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                
                full_text += text_delta
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_idx, 'delta': {'type': 'text_delta', 'text': text_delta}})}\n\n"
            
            # Usage from chunk
            u = chunk.get("usage", {})
            if u.get("prompt_tokens"):
                input_tokens = u["prompt_tokens"]
            if u.get("completion_tokens"):
                output_tokens = u["completion_tokens"]
        
        # Close last content block
        if full_text or reasoning_text:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_idx})}\n\n"
        
        # message_delta (stop reason)
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}})}\n\n"
        
        # message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        await response.aclose()
    
    return converter


# ── 数据管理 ──────────────────────────────────────
def load_keys():
    if KEYS_FILE.exists():
        return json.loads(KEYS_FILE.read_text())
    return {"admin_key": "sk-admin-luna2026", "keys": {}}

def save_keys(data):
    KEYS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

def get_key_info(api_key: str):
    data = load_keys()
    if api_key in data["keys"]:
        return data["keys"][api_key]
    return None

def record_usage(api_key: str, input_tokens: int, output_tokens: int, model: str):
    data = load_keys()
    if api_key not in data["keys"]:
        return
    key_info = data["keys"][api_key]
    key_info["usage"]["total_input"] += input_tokens
    key_info["usage"]["total_output"] += output_tokens
    key_info["usage"]["total_requests"] += 1
    key_info["usage"]["last_used"] = datetime.now(SGT).isoformat()
    # 按日统计
    today = datetime.now(SGT).strftime("%Y-%m-%d")
    daily = key_info["usage"].setdefault("daily", {})
    if today not in daily:
        daily[today] = {"input": 0, "output": 0, "requests": 0, "by_model": {}}
    daily[today]["input"] += input_tokens
    daily[today]["output"] += output_tokens
    daily[today]["requests"] += 1
    # 按模型统计（日维度，只记总量）
    by_model = daily[today].setdefault("by_model", {})
    if model not in by_model:
        by_model[model] = {"total": 0, "requests": 0}
    
    # Migration: if old format (input/output keys exist but no total), migrate it
    if "total" not in by_model[model]:
        by_model[model]["total"] = by_model[model].get("input", 0) + by_model[model].get("output", 0)
        # We can keep or remove input/output keys, keeping them is harmless but total is needed
        
    by_model[model]["total"] += input_tokens + output_tokens
    by_model[model]["requests"] += 1
    # 按小时+模型统计
    hour_key = datetime.now(SGT).strftime("%Y-%m-%d %H:00")
    hourly = key_info["usage"].setdefault("hourly", {})
    if hour_key not in hourly:
        hourly[hour_key] = {"by_model": {}}
    h_by_model = hourly[hour_key].setdefault("by_model", {})
    if model not in h_by_model:
        h_by_model[model] = {"total": 0, "requests": 0}
    h_by_model[model]["total"] += input_tokens + output_tokens
    h_by_model[model]["requests"] += 1
    # 全局按模型累计
    total_by_model = key_info["usage"].setdefault("total_by_model", {})
    if model not in total_by_model:
        total_by_model[model] = {"total": 0, "requests": 0}
    
    # Handle legacy format for global stats
    if "total" not in total_by_model[model]:
        total_by_model[model]["total"] = total_by_model[model].get("input", 0) + total_by_model[model].get("output", 0)
        
    total_by_model[model]["total"] += input_tokens + output_tokens
    total_by_model[model]["requests"] += 1
    save_keys(data)

# ── 鉴权 ──────────────────────────────────────────
def extract_api_key(request: Request) -> str:
    # 支持 x-api-key 和 Authorization: Bearer
    key = request.headers.get("x-api-key", "")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:]
    return key.strip()

def require_api_key(request: Request) -> dict:
    api_key = extract_api_key(request)
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    info = get_key_info(api_key)
    if info is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not info.get("enabled", True):
        raise HTTPException(status_code=403, detail="API key disabled")
    return {"key": api_key, **info}

def require_admin(request: Request):
    api_key = extract_api_key(request)
    data = load_keys()
    if api_key != data.get("admin_key"):
        raise HTTPException(status_code=403, detail="Admin key required")

# ── Fallback 逻辑 ────────────────────────────────

async def pick_tier(client: httpx.AsyncClient, requested_model: str) -> Optional[dict]:
    """根据额度选择最佳 tier"""
    config = load_fallback_config()
    if not config.get("enabled"):
        return None
    
    # Poll health (non-blocking, uses cache)
    await health_cache.poll(client)
    
    min_remaining = config.get("min_remaining_fraction", 0.05)
    tiers = config.get("tiers", [])
    
    # Find the tier matching the requested model
    requested_tier_idx = -1
    for i, tier in enumerate(tiers):
        if tier["model"] == requested_model:
            requested_tier_idx = i
            break
    
    # If requested model not in tiers, don't fallback
    if requested_tier_idx < 0:
        return None
    
    # Check if requested model has quota
    requested_tier = tiers[requested_tier_idx]
    if requested_tier["type"] == "antigravity":
        if health_cache.is_available(requested_tier.get("health_key", requested_model), min_remaining):
            return None  # Original model is fine, no fallback needed
    else:
        return None  # Non-antigravity models don't have health check
    
    # Requested model is low → find fallback
    for i, tier in enumerate(tiers):
        if i == requested_tier_idx:
            continue  # Skip the exhausted one
        
        if tier["type"] == "antigravity":
            if health_cache.is_available(tier.get("health_key", tier["model"]), min_remaining):
                log.info(f"⚡ Fallback: {requested_model} → {tier['model']} (quota low)")
                return tier
        else:
            # External tier (e.g., Kimi) is always available
            log.info(f"⚡ Fallback: {requested_model} → {tier['model']} (all antigravity exhausted)")
            return tier
    
    return None  # No fallback available


async def call_external_tier(tier: dict, body: dict, is_stream: bool, 
                              key_info: dict, original_model: str) -> StreamingResponse | JSONResponse:
    """调用外部 API（如 Kimi）"""
    openai_body = anthropic_to_openai(body)
    openai_body["model"] = tier["model"]
    
    # Respect max_tokens limit
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
            
            # Record usage
            usage = anthropic_resp.get("usage", {})
            record_usage(key_info["key"], 
                        usage.get("input_tokens", 0),
                        usage.get("output_tokens", 0),
                        tier["model"])
            
            return JSONResponse(content=anthropic_resp)


# ── FastAPI ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30))
    # Initial health poll
    try:
        await health_cache.poll(app.state.client)
        log.info(f"Initial health poll: {len(health_cache.models)} models")
    except:
        pass
    yield
    await app.state.client.aclose()

app = FastAPI(title="Luna API Proxy", lifespan=lifespan)

# ── 代理转发 ──────────────────────────────────────
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str, key_info: dict = Depends(require_api_key)):
    client: httpx.AsyncClient = request.app.state.client
    body = await request.body()
    
    # 构建上游请求头（去掉鉴权头，上游不需要）
    headers = {}
    for k, v in request.headers.items():
        if k.lower() not in ("host", "authorization", "x-api-key", "content-length"):
            headers[k] = v
    headers["x-api-key"] = "test"  # 上游固定 key
    
    # 检查是否是流式请求
    is_stream = False
    req_data = None
    if body:
        try:
            req_data = json.loads(body)
            is_stream = req_data.get("stream", False)
        except:
            pass
    
    # ── Fallback 检查 ──
    if req_data and path == "messages":
        requested_model = req_data.get("model", "")
        log.info(f"Incoming request for model: {requested_model}")
        request_start = time.time()
        fallback_tier = await pick_tier(client, requested_model)
        
        if fallback_tier:
            if fallback_tier["type"] != "antigravity":
                # External API (e.g., Kimi) — need format conversion
                return await call_external_tier(
                    fallback_tier, req_data, is_stream, key_info, requested_model)
            else:
                # Another Antigravity model — just swap model name
                req_data["model"] = fallback_tier["model"]
                body = json.dumps(req_data).encode()
                log.info(f"Swapped model in request: {requested_model} → {fallback_tier['model']}")
    
    url = f"/v1/{path}"
    
    if is_stream:
        # 流式转发
        upstream_req = client.build_request(
            method=request.method, url=url, headers=headers, content=body
        )
        upstream_resp = await client.send(upstream_req, stream=True)
        
        # Check for upstream error (429/503/exhausted) — reactive fallback
        should_fallback = False
        error_body = b""
        error_headers = upstream_resp.headers
        error_status = upstream_resp.status_code

        if req_data and path == "messages":
            if upstream_resp.status_code in (429, 503):
                should_fallback = True
            elif upstream_resp.status_code != 200:
                try:
                    # Read body to check for specific error text
                    error_body = await upstream_resp.aread()
                    error_text = error_body.decode("utf-8", errors="ignore").lower()
                    if any(x in error_text for x in ("exhausted", "capacity", "quota", "credit", "balance", "rate limit")):
                        should_fallback = True
                except:
                    pass

        if should_fallback:
            await upstream_resp.aclose()
            # Force health refresh and try fallback
            health_cache.last_poll = 0
            await health_cache.poll(client)
            
            config = load_fallback_config()
            tiers = config.get("tiers", [])
            requested_model = req_data.get("model", "")
            
            fallback_success = False
            for tier in tiers:
                if tier["model"] == requested_model:
                    continue
                
                # Check availability
                is_avail = False
                if tier["type"] == "antigravity":
                    is_avail = health_cache.is_available(tier.get("health_key", tier["model"]))
                else:
                    is_avail = True # External always avail
                
                if is_avail:
                    log.info(f"⚡ Reactive fallback ({error_status}): {requested_model} → {tier['model']}")
                    if tier["type"] == "antigravity":
                        req_data["model"] = tier["model"]
                        new_body = json.dumps(req_data).encode()
                        upstream_req = client.build_request(
                            method=request.method, url=url, headers=headers, content=new_body
                        )
                        upstream_resp = await client.send(upstream_req, stream=True)
                        if upstream_resp.status_code == 200:
                            fallback_success = True
                            break
                        else:
                            await upstream_resp.aclose() # Try next
                    else:
                        # External tier
                        return await call_external_tier(
                            tier, req_data, True, key_info, requested_model)
            
            if not fallback_success:
                # If all fallbacks failed, return original error (reconstructed)
                return JSONResponse(
                    content=json.loads(error_body) if error_body else {"error": "Service Unavailable"},
                    status_code=error_status
                )
        
        # If we read the body but didn't fallback, we need to return that error body
        if not should_fallback and error_body:
             return JSONResponse(
                content=json.loads(error_body) if error_body else {},
                status_code=error_status,
                headers=dict(error_headers)
            )

        input_tokens = 0
        output_tokens = 0
        model = ""
        STREAM_CHUNK_TIMEOUT = 180  # 3 minutes max silence between chunks
        
        async def stream_and_count():
            nonlocal input_tokens, output_tokens, model
            try:
                async for chunk in upstream_resp.aiter_bytes():
                    yield chunk
                    # 尝试从 SSE 中提取 usage
                    for line in chunk.decode("utf-8", errors="ignore").split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                d = json.loads(line[6:])
                                # message_start 里有 message.usage
                                msg = d.get("message", {})
                                if isinstance(msg, dict) and "usage" in msg:
                                    u = msg["usage"]
                                    input_tokens += u.get("input_tokens", 0) or 0
                                    output_tokens += u.get("output_tokens", 0) or 0
                                # message_delta 里有顶层 usage
                                if "usage" in d and "message" not in d:
                                    u = d["usage"]
                                    input_tokens += u.get("input_tokens", 0) or 0
                                    output_tokens += u.get("output_tokens", 0) or 0
                                if "model" in d:
                                    model = d["model"]
                                elif isinstance(msg, dict) and "model" in msg:
                                    model = msg["model"]
                            except:
                                pass
                await upstream_resp.aclose()
            except Exception as e:
                log.warning(f"Stream error for model={model or 'unknown'}: {e}")
                try:
                    await upstream_resp.aclose()
                except:
                    pass
            # 记录用量
            if input_tokens or output_tokens:
                record_usage(key_info["key"], input_tokens, output_tokens, model)
        
        return StreamingResponse(
            stream_and_count(),
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type", "text/event-stream")
        )
    else:
        # 非流式
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
                except:
                    pass

        if should_fallback:
            health_cache.last_poll = 0
            await health_cache.poll(client)
            
            config = load_fallback_config()
            tiers = config.get("tiers", [])
            requested_model = req_data.get("model", "")
            
            fallback_success = False
            for tier in tiers:
                if tier["model"] == requested_model:
                    continue
                
                # Check availability
                is_avail = False
                if tier["type"] == "antigravity":
                    is_avail = health_cache.is_available(tier.get("health_key", tier["model"]))
                else:
                    is_avail = True
                
                if is_avail:
                    if tier["type"] == "antigravity":
                        req_data["model"] = tier["model"]
                        new_body = json.dumps(req_data).encode()
                        resp = await client.request(
                            method=request.method, url=url, headers=headers, content=new_body
                        )
                        log.info(f"⚡ Reactive fallback ({resp.status_code}): {requested_model} → {tier['model']}")
                        if resp.status_code == 200:
                            fallback_success = True
                            break
                    else:
                        return await call_external_tier(
                            tier, req_data, False, key_info, requested_model)
        
        # 提取 usage
        try:
            resp_data = resp.json()
            usage = resp_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            model = resp_data.get("model", "")
            if input_tokens or output_tokens:
                record_usage(key_info["key"], input_tokens, output_tokens, model)
        except:
            pass
        
        return JSONResponse(
            content=resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": resp.text},
            status_code=resp.status_code
        )

# ── 管理接口 ──────────────────────────────────────
@app.post("/admin/keys")
async def create_key(request: Request):
    """创建新 API Key"""
    require_admin(request)
    body = await request.json()
    name = body.get("name", "unnamed")
    
    api_key = f"sk-{uuid.uuid4().hex[:24]}"
    data = load_keys()
    data["keys"][api_key] = {
        "name": name,
        "enabled": True,
        "created": datetime.now(SGT).isoformat(),
        "usage": {
            "total_input": 0,
            "total_output": 0,
            "total_requests": 0,
            "last_used": None,
            "daily": {}
        }
    }
    save_keys(data)
    return {"api_key": api_key, "name": name}

@app.get("/admin/keys")
async def list_keys(request: Request):
    """列出所有 API Key"""
    require_admin(request)
    data = load_keys()
    result = []
    for key, info in data["keys"].items():
        result.append({
            "key": f"{key[:8]}...{key[-4:]}",
            "full_key": key,
            "name": info["name"],
            "enabled": info["enabled"],
            "created": info["created"],
            "total_input": info["usage"]["total_input"],
            "total_output": info["usage"]["total_output"],
            "total_requests": info["usage"]["total_requests"],
            "last_used": info["usage"].get("last_used")
        })
    return {"keys": result}

@app.get("/admin/keys/{api_key}/usage")
async def key_usage(api_key: str, request: Request):
    """查看某个 Key 的详细用量"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    info = data["keys"][api_key]
    return {
        "name": info["name"],
        "enabled": info["enabled"],
        "usage": info["usage"]
    }

@app.post("/admin/keys/{api_key}/disable")
async def disable_key(api_key: str, request: Request):
    """停用 API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    data["keys"][api_key]["enabled"] = False
    save_keys(data)
    return {"status": "disabled", "name": data["keys"][api_key]["name"]}

@app.post("/admin/keys/{api_key}/enable")
async def enable_key(api_key: str, request: Request):
    """启用 API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    data["keys"][api_key]["enabled"] = True
    save_keys(data)
    return {"status": "enabled", "name": data["keys"][api_key]["name"]}

@app.delete("/admin/keys/{api_key}")
async def delete_key(api_key: str, request: Request):
    """删除 API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    name = data["keys"][api_key]["name"]
    del data["keys"][api_key]
    save_keys(data)
    return {"status": "deleted", "name": name}

@app.get("/admin/usage")
async def total_usage(request: Request):
    """总用量统计"""
    require_admin(request)
    data = load_keys()
    total_in = 0
    total_out = 0
    total_req = 0
    for info in data["keys"].values():
        total_in += info["usage"]["total_input"]
        total_out += info["usage"]["total_output"]
        total_req += info["usage"]["total_requests"]
    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_requests": total_req,
        "total_keys": len(data["keys"]),
        "active_keys": sum(1 for v in data["keys"].values() if v["enabled"])
    }

@app.get("/admin/usage/daily")
async def daily_usage(request: Request, date: str = None):
    """按日统计用量（含模型维度）"""
    require_admin(request)
    data = load_keys()

    daily_all = {}
    per_key = {}
    models_all = {}  # date -> model -> {input, output, requests}

    for key_id, info in data["keys"].items():
        name = info["name"]
        daily = info["usage"].get("daily", {})
        for d, stats in daily.items():
            if date and d != date:
                continue
            if d not in daily_all:
                daily_all[d] = {"input": 0, "output": 0, "requests": 0}
            daily_all[d]["input"] += stats["input"]
            daily_all[d]["output"] += stats["output"]
            daily_all[d]["requests"] += stats["requests"]
            if d not in per_key:
                per_key[d] = []
            per_key[d].append({
                "name": name,
                "input": stats["input"],
                "output": stats["output"],
                "requests": stats["requests"]
            })
            # Aggregate by_model
            for model, m_stats in stats.get("by_model", {}).items():
                if d not in models_all:
                    models_all[d] = {}
                if model not in models_all[d]:
                    models_all[d][model] = {"total": 0, "requests": 0}
                
                # Handle both formats
                total = m_stats.get("total", 0)
                if total == 0 and ("input" in m_stats or "output" in m_stats):
                    total = m_stats.get("input", 0) + m_stats.get("output", 0)
                
                models_all[d][model]["total"] += total
                models_all[d][model]["requests"] += m_stats.get("requests", 0)

    result = []
    for d in sorted(daily_all.keys()):
        result.append({
            "date": d,
            "total_input": daily_all[d]["input"],
            "total_output": daily_all[d]["output"],
            "total_requests": daily_all[d]["requests"],
            "by_model": dict(sorted(
                models_all.get(d, {}).items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )),
            "keys": sorted(per_key.get(d, []), key=lambda x: x["input"] + x["output"], reverse=True)
        })

    return {"days": result}

# ── Fallback 状态接口 ─────────────────────────────
@app.get("/admin/usage/hourly")
async def hourly_usage(request: Request, date: str = None):
    """按小时统计用量（含模型维度）"""
    require_admin(request)
    data = load_keys()
    
    hourly_all = {}
    for key_id, info in data["keys"].items():
        hourly = info["usage"].get("hourly", {})
        for h, stats in hourly.items():
            if date and not h.startswith(date):
                continue
            if h not in hourly_all:
                hourly_all[h] = {}
            for model, m_stats in stats.get("by_model", {}).items():
                if model not in hourly_all[h]:
                    hourly_all[h][model] = {"total": 0, "requests": 0}
                
                # Handle both old (input/output) and new (total) formats
                total = m_stats.get("total", 0)
                if total == 0 and ("input" in m_stats or "output" in m_stats):
                    total = m_stats.get("input", 0) + m_stats.get("output", 0)
                
                hourly_all[h][model]["total"] += total
                hourly_all[h][model]["requests"] += m_stats.get("requests", 0)
    
    result = []
    for h in sorted(hourly_all.keys()):
        result.append({"hour": h, "by_model": hourly_all[h]})
    
    return {"hours": result}

@app.get("/admin/fallback")
async def fallback_status(request: Request):
    """查看 fallback 状态和额度"""
    require_admin(request)
    config = load_fallback_config()
    await health_cache.poll(request.app.state.client)
    
    tiers_status = []
    for tier in config.get("tiers", []):
        status = {
            "name": tier["name"],
            "model": tier["model"],
            "type": tier["type"],
        }
        if tier["type"] == "antigravity":
            key = tier.get("health_key", tier["model"])
            remaining = health_cache.get_remaining(key)
            status["remaining"] = f"{remaining*100:.0f}%" if remaining >= 0 else "unknown"
            status["available"] = health_cache.is_available(key, config.get("min_remaining_fraction", 0.05))
        else:
            status["remaining"] = "unlimited"
            status["available"] = True
        tiers_status.append(status)
    
    return {
        "enabled": config.get("enabled", True),
        "tiers": tiers_status,
        "last_health_poll": datetime.fromtimestamp(health_cache.last_poll, SGT).isoformat() if health_cache.last_poll else None
    }

# ── OAuth callback for Lark calendar authorization ──
LARK_APP_ID = "cli_a90c3a6163785ed2"
LARK_APP_SECRET = "***LARK_SECRET_REMOVED***"
LARK_TOKEN_FILE = "/home/ubuntu/.openclaw/workspace/data/lark-user-token.json"

@app.post("/oauth/callback")
async def oauth_callback_post(request: Request):
    """Handle Lark challenge verification AND card action callbacks."""
    body = await request.json()
    # Challenge verification
    if "challenge" in body:
        return {"challenge": body["challenge"]}
    # Card action callback: forward to OpenClaw webhook
    event_type = ""
    if isinstance(body.get("header"), dict):
        event_type = body["header"].get("event_type", "")
    if event_type.startswith("card.action.trigger"):
        try:
            client: httpx.AsyncClient = request.app.state.client
            resp = await client.post(
                "http://127.0.0.1:18789/webhook/lark",
                json=body,
                timeout=5
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "ok"}

@app.get("/oauth/callback")
async def oauth_callback(code: str = None, state: str = None):
    if not code:
        return {"status": "ok"}
    
    import urllib.request
    # Get app_access_token
    req = urllib.request.Request(
        "https://open.larksuite.com/open-apis/auth/v3/app_access_token/internal",
        data=json.dumps({"app_id": LARK_APP_ID, "app_secret": LARK_APP_SECRET}).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    app_token = json.loads(resp.read()).get("app_access_token", "")
    
    # Exchange code for user_access_token
    req = urllib.request.Request(
        "https://open.larksuite.com/open-apis/authen/v1/oidc/access_token",
        data=json.dumps({"grant_type": "authorization_code", "code": code}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {app_token}"})
    resp = urllib.request.urlopen(req)
    token_data = json.loads(resp.read())
    
    if token_data.get("code") == 0:
        os.makedirs(os.path.dirname(LARK_TOKEN_FILE), exist_ok=True)
        with open(LARK_TOKEN_FILE, "w") as f:
            json.dump(token_data.get("data", {}), f, indent=2)
        from fastapi.responses import HTMLResponse
        return HTMLResponse("✅ 授权成功！Luna 已获得日历访问权限。你可以关闭这个页面了。")
    else:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(f"❌ 授权失败：{json.dumps(token_data, ensure_ascii=False)}")

@app.get("/health")
async def health():
    config = load_fallback_config()
    return {
        "status": "ok",
        "upstream": UPSTREAM,
        "fallback_enabled": config.get("enabled", True),
        "tiers": len(config.get("tiers", []))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
