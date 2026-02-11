#!/bin/bash
# Usage: ./scripts/deploy.sh [dev|prod]
# Deploys the latest code from main branch
set -euo pipefail

ENV=${1:-dev}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

case "$ENV" in
  dev)
    SERVICE="api-proxy-dev"
    PORT=8181
    echo "🔄 Deploying to DEV environment..."
    ;;
  prod|production)
    SERVICE="api-proxy"
    PORT=8180
    echo "🔄 Deploying to PRODUCTION environment..."
    ;;
  *)
    echo "❌ Unknown environment: $ENV"
    echo "Usage: $0 [dev|prod]"
    exit 1
    ;;
esac

# Pull latest code
echo "📥 Pulling latest code..."
git pull origin main

# Syntax check before restarting
echo "🔍 Syntax check..."
python3 -m py_compile server.py
echo "  ✅ Syntax OK"

# Restart service
echo "🔄 Restarting $SERVICE..."
sudo systemctl restart "$SERVICE"
sleep 3

# Check service status
if sudo systemctl is-active --quiet "$SERVICE"; then
    echo "  ✅ $SERVICE is running"
else
    echo "  ❌ $SERVICE failed to start!"
    sudo journalctl -u "$SERVICE" --no-pager -n 20
    exit 1
fi

# Run smoke test
echo "🧪 Running smoke tests on port $PORT..."
bash "$SCRIPT_DIR/smoke-test.sh" "$PORT"

echo ""
echo "✅ Deployment to $ENV complete!"
