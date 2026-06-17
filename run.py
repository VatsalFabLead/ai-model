#!/usr/bin/env python3
"""Local dev server entrypoint."""

import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when run directly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

import uvicorn

from app.config import get_settings


def main() -> None:
  settings = get_settings()
  uvicorn.run(
    "app.main:app",
    host=settings.host,
    port=settings.port,
    reload=settings.reload and not settings.is_production,
    timeout_keep_alive=settings.request_timeout_seconds,
  )


if __name__ == "__main__":
  main()
