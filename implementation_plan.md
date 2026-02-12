# Implementation Plan: Dual Native Format Support

## Goal
Support native pass-through for both Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) formats, removing forced conversions when unnecessary.

## Changes

### 1. Modify `src/app.py`

#### A. Refactor `call_external_tier`
Update signature to accept `client_format` ("anthropic" or "openai").

Logic:
- If `client_format == "anthropic"`:
  - Request: `anthropic_to_openai(body)`
  - Response: `openai_to_anthropic(resp)` / `openai_stream_to_anthropic_stream(resp)`
- If `client_format == "openai"`:
  - Request: Use `body` directly (ensure `model` is updated to tier model).
  - Response: Return JSON/Stream directly.

#### B. Split Routes
Remove generic catch-all and create specific endpoints:

```python
@app.post("/v1/messages")
async def post_messages(request: Request, key_info: dict = Depends(require_api_key)):
    # client_format = "anthropic"
    # ... logic ...

@app.post("/v1/chat/completions")
async def post_chat_completions(request: Request, key_info: dict = Depends(require_api_key)):
    # client_format = "openai"
    # ... logic ...
```

#### C. Refactor Shared Logic
Extract common proxy/fallback logic into a `_handle_request` helper that accepts `client_format`.

### 2. Verify `src/proxy.py`
Ensure `anthropic_to_openai` handles any edge cases if invoked, but primary fix is avoiding invocation for OpenAI-native requests.

### 3. Verification
- Test `/v1/messages` -> Fallback (should convert).
- Test `/v1/chat/completions` -> Fallback (should NOT convert).
