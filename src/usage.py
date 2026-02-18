"""Usage tracking and recording"""
from datetime import datetime

from .auth import load_keys, save_keys
from .config import SGT


def record_usage(api_key: str, input_tokens: int, output_tokens: int, model: str,
                 session_id: str = ""):
    """Record usage for an API key, optionally tagged with a session_id."""
    data = load_keys()
    if api_key not in data["keys"]:
        return
    key_info = data["keys"][api_key]
    key_info["usage"]["total_input"] += input_tokens
    key_info["usage"]["total_output"] += output_tokens
    key_info["usage"]["total_requests"] += 1
    key_info["usage"]["last_used"] = datetime.now(SGT).isoformat()

    # Daily stats
    today = datetime.now(SGT).strftime("%Y-%m-%d")
    daily = key_info["usage"].setdefault("daily", {})
    if today not in daily:
        daily[today] = {"input": 0, "output": 0, "requests": 0, "by_model": {}}
    daily[today]["input"] += input_tokens
    daily[today]["output"] += output_tokens
    daily[today]["requests"] += 1

    # By-model (daily)
    by_model = daily[today].setdefault("by_model", {})
    if model not in by_model:
        by_model[model] = {"total": 0, "requests": 0}

    # Migration: if old format (input/output keys exist but no total), migrate it
    if "total" not in by_model[model]:
        by_model[model]["total"] = by_model[model].get("input", 0) + by_model[model].get("output", 0)

    by_model[model]["total"] += input_tokens + output_tokens
    by_model[model]["requests"] += 1

    # Hourly + model stats
    hour_key = datetime.now(SGT).strftime("%Y-%m-%d %H:00")
    hourly = key_info["usage"].setdefault("hourly", {})
    if hour_key not in hourly:
        hourly[hour_key] = {"by_model": {}}
    h_by_model = hourly[hour_key].setdefault("by_model", {})
    if model not in h_by_model:
        h_by_model[model] = {"total": 0, "requests": 0}
    h_by_model[model]["total"] += input_tokens + output_tokens
    h_by_model[model]["requests"] += 1

    # Global by-model totals
    total_by_model = key_info["usage"].setdefault("total_by_model", {})
    if model not in total_by_model:
        total_by_model[model] = {"total": 0, "requests": 0}

    # Handle legacy format for global stats
    if "total" not in total_by_model[model]:
        total_by_model[model]["total"] = total_by_model[model].get("input", 0) + total_by_model[model].get("output", 0)

    total_by_model[model]["total"] += input_tokens + output_tokens
    total_by_model[model]["requests"] += 1

    # Per-session tracking (if session_id provided)
    if session_id:
        by_session = key_info["usage"].setdefault("by_session", {})
        if session_id not in by_session:
            by_session[session_id] = {
                "input": 0, "output": 0, "requests": 0,
                "first_seen": datetime.now(SGT).isoformat(),
            }
        by_session[session_id]["input"] += input_tokens
        by_session[session_id]["output"] += output_tokens
        by_session[session_id]["requests"] += 1
        by_session[session_id]["last_seen"] = datetime.now(SGT).isoformat()

        # Prune old sessions (keep last 200)
        if len(by_session) > 200:
            sorted_sessions = sorted(by_session.items(),
                                     key=lambda x: x[1].get("last_seen", ""),
                                     reverse=True)
            key_info["usage"]["by_session"] = dict(sorted_sessions[:200])

    save_keys(data)


def get_session_usage(api_key: str, session_id: str) -> dict | None:
    """Get usage for a specific session."""
    data = load_keys()
    if api_key not in data["keys"]:
        return None
    by_session = data["keys"][api_key].get("usage", {}).get("by_session", {})
    return by_session.get(session_id)
