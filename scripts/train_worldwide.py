#!/usr/bin/env python3
"""Train Nexus worldwide — 100% custom, no GPT/Claude/Gemini.

Pipeline:
  1. Import assistant profile (capabilities, languages, categories, human types)
  2. Optionally import free public dataset samples into KB + corpus
  3. Train transformer weights (inference-only at chat time)

Usage:
  python scripts/train_worldwide.py
  python scripts/train_worldwide.py --epochs 12 --seq-len 128
  python scripts/train_worldwide.py --skip-public-datasets
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
  print("\n>>", " ".join(cmd))
  subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
  parser = argparse.ArgumentParser(description="Worldwide Nexus training pipeline")
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--skip-public-datasets", action="store_true")
  parser.add_argument("--public-samples", type=int, default=10)
  args = parser.parse_args()

  py = sys.executable
  corpus = PROJECT_ROOT / "data" / "corpus.txt"

  run([py, str(PROJECT_ROOT / "scripts" / "import_chat_training.py"), "--build-corpus"])

  if not args.skip_public_datasets:
    run([
      py,
      str(PROJECT_ROOT / "scripts" / "import_public_datasets.py"),
      "--datasets",
      "gooaq,wikidata,gutenberg,arxiv,stackexchange,c4,datacommons",
      "--samples", str(args.public_samples),
      "--build-corpus",
    ])

  run([
    py,
    str(PROJECT_ROOT / "scripts" / "train.py"),
    "--corpus", str(corpus),
    "--epochs", str(args.epochs),
    "--seq-len", str(args.seq_len),
    "--lr", str(args.lr),
  ])

  print("\nDone. Weights saved. Restart server: python run.py")
  print("Chat uses inference only — tools supply context; the model writes every answer.")


if __name__ == "__main__":
  main()
