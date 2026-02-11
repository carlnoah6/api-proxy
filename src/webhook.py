"""GitHub webhook handler for CI/CD notifications"""
import hashlib
import hmac
import json
import subprocess
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .config import log

router = APIRouter()

SECRET_FILE = Path(__file__).resolve().parent.parent / "webhook_secret.txt"
LARK_SCRIPT = "/home/ubuntu/.openclaw/workspace/scripts/lark-send-message.sh"
LARK_CHAT_ID = "oc_680d9c843e6a0ad501de9299a97f3a7e"

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
    # Expected format: sha256=<hex_digest>
    if not signature_header.startswith("sha256="):
        return False
    expected = signature_header[7:]
    computed = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, expected)


def _send_lark(message: str):
    """Send a message via Lark script"""
    try:
        result = subprocess.run(
            [LARK_SCRIPT, LARK_CHAT_ID, message],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            log.error(f"Lark send failed: {result.stderr}")
        else:
            log.info(f"Lark notification sent: {result.stdout.strip()}")
    except Exception as e:
        log.error(f"Lark send error: {e}")


def _is_deploy_workflow(name: str) -> bool:
    """Check if workflow name indicates a deployment"""
    return name.lower().strip() in DEPLOY_WORKFLOWS


def _format_message(data: dict) -> str | None:
    """
    Format webhook payload into a Lark notification message.
    Returns None if no notification should be sent.
    """
    workflow_run = data.get("workflow_run", {})
    repo = workflow_run.get("repository", {}).get("full_name", "unknown")
    workflow_name = workflow_run.get("name", "unknown")
    conclusion = workflow_run.get("conclusion", "unknown")
    run_url = workflow_run.get("html_url", "")
    branch = workflow_run.get("head_branch", "unknown")
    commit_msg = workflow_run.get("head_commit", {}).get("message", "").split("\n")[0] if workflow_run.get("head_commit") else ""

    # Extract PR info if available
    pr_info = ""
    pull_requests = workflow_run.get("pull_requests", [])
    if pull_requests:
        pr = pull_requests[0]
        pr_number = pr.get("number", "")
        pr_info = f"\nPR: #{pr_number}"

    is_deploy = _is_deploy_workflow(workflow_name)

    if conclusion == "success" and is_deploy:
        # Deploy succeeded — notify
        emoji = "🚀"
        status = "Deploy Succeeded"
    elif conclusion in ("failure", "cancelled", "timed_out"):
        # Any failure — notify
        emoji = "❌" if conclusion == "failure" else "⚠️"
        status = f"CI {conclusion.replace('_', ' ').title()}"
    elif conclusion == "success":
        # Regular CI success — skip (too noisy)
        log.info(f"Skipping notification: {repo} / {workflow_name} succeeded (not deploy)")
        return None
    else:
        # Other conclusions (action_required, stale, etc.) — skip
        log.info(f"Skipping notification: {repo} / {workflow_name} conclusion={conclusion}")
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
    # Read raw body for signature verification
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

    # Parse event type
    event_type = request.headers.get("x-github-event", "")
    log.info(f"GitHub webhook received: event={event_type}")

    # Handle ping (sent when webhook is first created)
    if event_type == "ping":
        return JSONResponse(content={"ok": True, "msg": "pong"})

    # Only process workflow_run events
    if event_type != "workflow_run":
        return JSONResponse(content={"ok": True, "msg": f"ignored event: {event_type}"})

    # Parse payload
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)

    action = data.get("action", "")
    log.info(f"GitHub webhook: workflow_run action={action}")

    # Only process completed runs
    if action != "completed":
        return JSONResponse(content={"ok": True, "msg": f"ignored action: {action}"})

    # Format and send notification
    message = _format_message(data)
    if message:
        _send_lark(message)

    return JSONResponse(content={"ok": True})
