#!/usr/bin/env python3
"""
Luna API Proxy - Entry point
Multi-tenant API proxy with auth, usage tracking, and smart fallback.

This is a thin entry point that imports the modular application.
All logic lives in the src/ package.
"""

from src.app import app  # noqa: F401
from src.config import LISTEN_PORT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT)
