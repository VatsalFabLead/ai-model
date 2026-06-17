#!/usr/bin/env python3
"""Quick local test: load trained weights and chat from the terminal."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.engine.inference import GenerationConfig, InferenceEngine


def main() -> None:
  settings = get_settings()
  engine = InferenceEngine(settings)
  engine.load()

  prompts = sys.argv[1:] or [
    "Hello",
    "Who are you?",
    "What is your name?",
    "Can you help me with Python?",
    "Are you free?",
  ]
  cfg = GenerationConfig(
    max_new_tokens=40, temperature=0.0, top_k=0, top_p=1.0,
    repetition_penalty=1.3, no_repeat_ngram=2,
  )
  for q in prompts:
    messages = [{"role": "user", "content": q}]
    prompt = engine.format_chat_prompt(messages)
    answer = engine.generate(prompt, cfg)
    print(f"User: {q}")
    print(f"Nexus: {answer.strip()}")
    print("-" * 50)


if __name__ == "__main__":
  main()
