#!/usr/bin/env python3
"""Entry point — run with: python run.py"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use platform port if available

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,   # Disable reload in production
        log_level="info",
    )