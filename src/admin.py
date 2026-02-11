"""Admin endpoints for key management, usage stats, and fallback status"""
import json
import os
from datetime import datetime

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import create_api_key, load_keys, require_admin, save_keys
from .config import LARK_APP_ID, LARK_APP_SECRET, LARK_TOKEN_FILE, SGT
from .fallback import health_cache, load_fallback_config

# ── Key Management ──


async def admin_create_key(request: Request):
    """Create a new API Key"""
    require_admin(request)
    body = await request.json()
    name = body.get("name", "unnamed")
    api_key, _ = create_api_key(name)
    return {"api_key": api_key, "name": name}


async def admin_list_keys(request: Request):
    """List all API Keys"""
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


async def admin_key_usage(api_key: str, request: Request):
    """View detailed usage for a specific key"""
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


async def admin_disable_key(api_key: str, request: Request):
    """Disable an API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    data["keys"][api_key]["enabled"] = False
    save_keys(data)
    return {"status": "disabled", "name": data["keys"][api_key]["name"]}


async def admin_enable_key(api_key: str, request: Request):
    """Enable an API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    data["keys"][api_key]["enabled"] = True
    save_keys(data)
    return {"status": "enabled", "name": data["keys"][api_key]["name"]}


async def admin_delete_key(api_key: str, request: Request):
    """Delete an API Key"""
    require_admin(request)
    data = load_keys()
    if api_key not in data["keys"]:
        raise HTTPException(status_code=404, detail="Key not found")
    name = data["keys"][api_key]["name"]
    del data["keys"][api_key]
    save_keys(data)
    return {"status": "deleted", "name": name}


# ── Usage Stats ──


async def admin_total_usage(request: Request):
    """Total usage across all keys"""
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


async def admin_daily_usage(request: Request, date: str = None):
    """Daily usage stats with model breakdown"""
    require_admin(request)
    data = load_keys()

    daily_all = {}
    per_key = {}
    models_all = {}

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
            for model, m_stats in stats.get("by_model", {}).items():
                if d not in models_all:
                    models_all[d] = {}
                if model not in models_all[d]:
                    models_all[d][model] = {"total": 0, "requests": 0}

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


async def admin_hourly_usage(request: Request, date: str = None):
    """Hourly usage stats with model breakdown"""
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

                total = m_stats.get("total", 0)
                if total == 0 and ("input" in m_stats or "output" in m_stats):
                    total = m_stats.get("input", 0) + m_stats.get("output", 0)

                hourly_all[h][model]["total"] += total
                hourly_all[h][model]["requests"] += m_stats.get("requests", 0)

    result = []
    for h in sorted(hourly_all.keys()):
        result.append({"hour": h, "by_model": hourly_all[h]})

    return {"hours": result}


# ── Fallback Status ──


async def admin_fallback_status(request: Request):
    """View fallback status and quota info"""
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


# ── OAuth / Lark callbacks ──


async def oauth_callback_post(request: Request):
    """Handle Lark challenge verification AND card action callbacks."""
    body = await request.json()
    if "challenge" in body:
        return {"challenge": body["challenge"]}

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


async def oauth_callback_get(code: str = None, state: str = None):
    """Handle OAuth callback from Lark"""
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
        return HTMLResponse("✅ Authorization successful! Calendar access has been granted. You can close this page now.")
    else:
        from fastapi.responses import HTMLResponse
        return HTMLResponse(f"❌ Authorization failed: {json.dumps(token_data, ensure_ascii=False)}")
