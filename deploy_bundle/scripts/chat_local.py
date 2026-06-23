#!/usr/bin/env python3
"""Quick local test: chat with the full custom stack (knowledge engine + model)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

try:
  sys.stdout.reconfigure(encoding="utf-8")
except Exception:
  pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.services.registry import ProviderRegistry


async def run(prompts: list[str]) -> None:
  settings = get_settings()
  registry = ProviderRegistry(settings)
  await registry.startup()
  provider = registry.provider
  print(f"Backend: {settings.model_backend}")
  kb = getattr(provider, "_kb", None)
  print(f"Knowledge base entries: {kb.size if kb else 0}")
  print("=" * 60)
  for q in prompts:
    answer = await provider.chat([{"role": "user", "content": q}])
    print(f"User:  {q}")
    print(f"Nexus: {answer}")
    print("-" * 60)
  await registry.shutdown()


def main() -> None:
  prompts = sys.argv[1:] or [
    "Hello",
    "I want to make Flutter CV",
    "What is the capital of Japan?",
    "what is machine learning",
    "namaste",
    "Tell me about quantum entanglement",
  ]
  asyncio.run(run(prompts))


if __name__ == "__main__":
  main()
