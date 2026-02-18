"""Admin endpoints for key management, usage stats, and model status"""
import json
import os
import urllib.request

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse

from .auth import create_api_key, load_keys, require_admin, save_keys
from .usage import get_session_usage
from .config import LARK_APP_ID, LARK_APP_SECRET, LARK_TOKEN_FILE, get_known_models, get_providers

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
    """Daily usage stats with model breakdown (multi-tenant)

    Returns daily usage aggregated by tenant (API key) and by model.
    """
    require_admin(request)
    data = load_keys()

    daily_all = {}
    per_key = {}  # Tenant-level data
    models_all = {}  # Global model stats
    models_by_key = {}  # Per-tenant model stats: {date: {key_name: {model: {...}}}}

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

            # Per-tenant model stats
            key_models = {}
            for model, m_stats in stats.get("by_model", {}).items():
                total = m_stats.get("total", 0)
                if total == 0 and ("input" in m_stats or "output" in m_stats):
                    total = m_stats.get("input", 0) + m_stats.get("output", 0)

                key_models[model] = {
                    "total": total,
                    "requests": m_stats.get("requests", 0)
                }

                # Global model stats
                if d not in models_all:
                    models_all[d] = {}
                if model not in models_all[d]:
                    models_all[d][model] = {"total": 0, "requests": 0}
                models_all[d][model]["total"] += total
                models_all[d][model]["requests"] += m_stats.get("requests", 0)

            per_key[d].append({
                "name": name,
                "input": stats["input"],
                "output": stats["output"],
                "requests": stats["requests"],
                "by_model": key_models
            })

            # Store per-tenant model stats
            if d not in models_by_key:
                models_by_key[d] = {}
            models_by_key[d][name] = key_models

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
            "by_tenant": models_by_key.get(d, {}),  # New: per-tenant model breakdown
            "keys": sorted(per_key.get(d, []), key=lambda x: x["input"] + x["output"], reverse=True)
        })

    return {"days": result}


async def admin_hourly_usage(request: Request, date: str = None):
    """Hourly usage stats with model breakdown (multi-tenant)

    Returns hourly usage aggregated by tenant (API key) and by model.
    """
    require_admin(request)
    data = load_keys()

    hourly_all = {}  # {hour: {model: {...}}}
    hourly_by_tenant = {}  # {hour: {tenant: {model: {...}}}}

    for key_id, info in data["keys"].items():
        name = info["name"]
        hourly = info["usage"].get("hourly", {})
        for h, stats in hourly.items():
            if date and not h.startswith(date):
                continue

            if h not in hourly_all:
                hourly_all[h] = {}
            if h not in hourly_by_tenant:
                hourly_by_tenant[h] = {}
            if name not in hourly_by_tenant[h]:
                hourly_by_tenant[h][name] = {}

            for model, m_stats in stats.get("by_model", {}).items():
                total = m_stats.get("total", 0)
                if total == 0 and ("input" in m_stats or "output" in m_stats):
                    total = m_stats.get("input", 0) + m_stats.get("output", 0)

                requests = m_stats.get("requests", 0)

                # Global model stats
                if model not in hourly_all[h]:
                    hourly_all[h][model] = {"total": 0, "requests": 0}
                hourly_all[h][model]["total"] += total
                hourly_all[h][model]["requests"] += requests

                # Per-tenant model stats
                if model not in hourly_by_tenant[h][name]:
                    hourly_by_tenant[h][name][model] = {"total": 0, "requests": 0}
                hourly_by_tenant[h][name][model]["total"] += total
                hourly_by_tenant[h][name][model]["requests"] += requests

    result = []
    for h in sorted(hourly_all.keys()):
        result.append({
            "hour": h,
            "by_model": hourly_all[h],
            "by_tenant": hourly_by_tenant.get(h, {})  # New: per-tenant breakdown
        })

    return {"hours": result}


# ── Model Status ──


async def admin_models_status(request: Request):
    """View registered models and their configuration"""
    require_admin(request)
    providers = get_providers()
    known = get_known_models()
    plist = []
    for pid, p in providers.items():
        plist.append({"id": pid, "name": p.get("name", pid), "has_api_key": bool(p.get("api_key"))})
    mlist = [{"id": m["id"], "provider": m["provider"]} for m in known]
    return {"providers": plist, "known_models": mlist}


# ── Session Usage ──


async def admin_session_usage(api_key: str, session_id: str, request: Request):
    """Get usage for a specific session under an API key."""
    require_admin(request)
    usage = get_session_usage(api_key, session_id)
    if usage is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "usage": usage}


# ── OAuth / Lark callbacks ──


async def oauth_callback_get(code: str = None, state: str = None):
    """Handle OAuth callback from Lark"""
    if not code:
        return {"status": "ok"}

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
        return HTMLResponse("✅ Authorization successful! Calendar access has been granted. You can close this page now.")
    else:
        return HTMLResponse(f"❌ Authorization failed: {json.dumps(token_data, ensure_ascii=False)}")
