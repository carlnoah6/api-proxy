"""GitHub webhook handler for CI/CD notifications.

Instead of sending Lark messages directly (which causes cross-posting),
saves events to files for Luna to process during heartbeat.
"""
import hashlib
import hmac
import json
import subprocess
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .config import log

router = APIRouter()

SECRET_FILE = Path(__file__).resolve().parent.parent / "webhook_secret.txt"
EVENT_DIR = Path("/home/ubuntu/.openclaw/workspace/data/ci-events")

# Workflow names that count as "deploy" (case-insensitive)
DEPLOY_WORKFLOWS = {"deploy", "deployment", "release"}


def _load_secret() -> str:
    """Load webhook secret from file"""
    try:
        return SECRET_FILE.read_text().strip()
    except Exception as e:
        log.error(f"Failed to load webhook secret: {e}")
        return ""


def _verify_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    """Verify GitHub HMAC-SHA256 signature"""
    if not signature_header or not secret:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = signature_header[7:]
    computed = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, expected)


def _save_event(message: str, data: dict):
    """Save CI event to file for Luna to process, then trigger wake."""
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    workflow_run = data.get("workflow_run", {})
    event_data = {
        "message": message,
        "repo": workflow_run.get("repository", {}).get("full_name", ""),
        "workflow": workflow_run.get("name", ""),
        "conclusion": workflow_run.get("conclusion", ""),
        "run_url": workflow_run.get("html_url", ""),
        "branch": workflow_run.get("head_branch", ""),
        "timestamp": time.time(),
    }
    event_file = EVENT_DIR / f"{int(time.time() * 1000)}.json"
    event_file.write_text(json.dumps(event_data, indent=2))
    log.info(f"CI event saved: {event_file.name}")
    # Trigger OpenClaw wake so Luna picks it up quickly
    try:
        subprocess.run(
            ["openclaw", "cron", "add", "--wake", "now"],
            capture_output=True, timeout=10
        )
    except Exception as e:
        log.warning(f"Failed to trigger wake: {e}")


def _is_deploy_workflow(name: str) -> bool:
    """Check if workflow name indicates a deployment"""
    return name.lower().strip() in DEPLOY_WORKFLOWS


def _format_message(data: dict) -> str | None:
    """
    Format webhook payload into a notification message.
    Returns None if no notification should be sent.
    """
    workflow_run = data.get("workflow_run", {})
    repo = workflow_run.get("repository", {}).get("full_name", "unknown")
    workflow_name = workflow_run.get("name", "unknown")
    conclusion = workflow_run.get("conclusion", "unknown")
    run_url = workflow_run.get("html_url", "")
    branch = workflow_run.get("head_branch", "unknown")
    commit_msg = (
        workflow_run.get("head_commit", {}).get("message", "").split("\n")[0]
        if workflow_run.get("head_commit") else ""
    )

    pr_info = ""
    pull_requests = workflow_run.get("pull_requests", [])
    if pull_requests:
        pr_number = pull_requests[0].get("number", "")
        pr_info = f"\nPR: #{pr_number}"

    is_deploy = _is_deploy_workflow(workflow_name)

    if conclusion == "success" and is_deploy:
        emoji = "🚀"
        status = "Deploy Succeeded"
    elif conclusion in ("failure", "cancelled", "timed_out"):
        emoji = "❌" if conclusion == "failure" else "⚠️"
        status = f"CI {conclusion.replace('_', ' ').title()}"
    elif conclusion == "success":
        # Regular CI success — skip (too noisy)
        log.info(f"Skipping: {repo} / {workflow_name} succeeded (not deploy)")
        return None
    else:
        log.info(f"Skipping: {repo} / {workflow_name} conclusion={conclusion}")
        return None

    lines = [
        f"{emoji} {status}",
        f"Repo: {repo}",
        f"Workflow: {workflow_name}",
        f"Branch: {branch}",
    ]
    if commit_msg:
        lines.append(f"Commit: {commit_msg}")
    if pr_info:
        lines.append(pr_info)
    lines.append(f"Link: {run_url}")

    return "\n".join(lines)


@router.post("/webhook/github")
async def github_webhook(request: Request):
    """Handle GitHub webhook events"""
    body = await request.body()

    # Verify signature
    secret = _load_secret()
    signature = request.headers.get("x-hub-signature-256", "")
    if not _verify_signature(body, signature, secret):
        log.warning("GitHub webhook: invalid signature")
        return JSONResponse(
            content={"error": "Invalid signature"},
            status_code=401
        )

    event_type = request.headers.get("x-github-event", "")
    log.info(f"GitHub webhook received: event={event_type}")

    if event_type == "ping":
        return JSONResponse(content={"ok": True, "msg": "pong"})

    if event_type != "workflow_run":
        return JSONResponse(content={"ok": True, "msg": f"ignored event: {event_type}"})

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    action = data.get("action", "")
    log.info(f"GitHub webhook: workflow_run action={action}")

    if action != "completed":
        return JSONResponse(content={"ok": True, "msg": f"ignored action: {action}"})

    # Save event to file for Luna to process (no direct Lark messages!)
    message = _format_message(data)
    if message:
        _save_event(message, data)

    return JSONResponse(content={"ok": True})
