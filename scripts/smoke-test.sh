#!/bin/bash
# Smoke test for API Proxy
# Usage: ./scripts/smoke-test.sh [PORT]
# Set SMOKE_TEST_KEY env var for auth tests (default: sk-dev-test-key)
set -euo pipefail

PORT=${1:-8181}
BASE="http://localhost:$PORT"
TEST_KEY="${SMOKE_TEST_KEY:-sk-dev-test-key}"
PASSED=0
FAILED=0

pass() { PASSED=$((PASSED + 1)); echo "  ✅ $1"; }
fail() { FAILED=$((FAILED + 1)); echo "  ❌ $1: $2"; }

echo "🧪 Smoke testing $BASE ..."
echo ""

# 1. Health check
echo "1️⃣  Health check"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/health" 2>/dev/null)
if [ "$HEALTH" = "200" ]; then
    pass "GET /health → 200"
else
    fail "GET /health" "got $HEALTH"
fi

# 2. Auth test - no key should fail
echo "2️⃣  Auth test (no key → 401/403)"
AUTH=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
if [ "$AUTH" = "401" ] || [ "$AUTH" = "403" ]; then
    pass "No API key → $AUTH"
else
    fail "No API key" "expected 401/403, got $AUTH"
fi

# 3. Auth test - invalid key should fail
echo "3️⃣  Auth test (invalid key → 401/403)"
BADKEY=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "x-api-key: sk-invalid-key-12345" \
    -d '{"model":"test","messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
if [ "$BADKEY" = "401" ] || [ "$BADKEY" = "403" ]; then
    pass "Invalid API key → $BADKEY"
else
    fail "Invalid API key" "expected 401/403, got $BADKEY"
fi

# 4. Non-streaming request (Anthropic /v1/messages format)
echo "4️⃣  Non-streaming request (gemini-3-flash)"
RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $TEST_KEY" \
    -d '{"model":"gemini-3-flash","messages":[{"role":"user","content":"Say hello in exactly 3 words"}],"max_tokens":50}' 2>/dev/null)
HTTP_CODE=$(echo "$RESP" | tail -1)
BODY=$(echo "$RESP" | head -n -1)
if [ "$HTTP_CODE" = "200" ]; then
    pass "Non-streaming gemini-3-flash → 200"
else
    # May fail if key not in this env's keys.json - that's OK for prod smoke test
    if [ "$HTTP_CODE" = "401" ] || [ "$HTTP_CODE" = "403" ]; then
        echo "  ⚠️  Skipped (test key not in this env)"
    else
        fail "Non-streaming request" "got $HTTP_CODE"
    fi
fi

# 5. Streaming request (Anthropic /v1/messages format)
echo "5️⃣  Streaming request (gemini-3-flash)"
STREAM=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $TEST_KEY" \
    -d '{"model":"gemini-3-flash","messages":[{"role":"user","content":"Say hi"}],"stream":true,"max_tokens":20}' 2>/dev/null)
if [ "$STREAM" = "200" ]; then
    pass "Streaming gemini-3-flash → 200"
else
    if [ "$STREAM" = "401" ] || [ "$STREAM" = "403" ]; then
        echo "  ⚠️  Skipped (test key not in this env)"
    else
        fail "Streaming request" "got $STREAM"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: $PASSED passed, $FAILED failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
