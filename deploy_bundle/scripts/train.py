#!/usr/bin/env python3
"""Train your fully custom model on local data — free, no third-party models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.engine.training import train_model
from app.engine.transformer import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
  settings = get_settings()
  parser = argparse.ArgumentParser(description="Train custom Nexus model")
  parser.add_argument(
    "--corpus",
    type=Path,
    default=PROJECT_ROOT / "data" / "corpus.txt",
    help="Path to training text file",
  )
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--seq-len", type=int, default=64)
  args = parser.parse_args()

  if not args.corpus.exists():
    args.corpus.parent.mkdir(parents=True, exist_ok=True)
    args.corpus.write_text(
      "\n".join(
        [
          "You are a helpful AI assistant built as a custom local model.",
          "This model runs entirely on your own server without third-party APIs.",
          "User: Hello, who are you?",
          "Assistant: I am Nexus, your custom local AI assistant.",
          "User: How do you work?",
          "Assistant: I use a custom transformer trained on your own data, running free on your server.",
        ]
      ),
      encoding="utf-8",
    )
    print(f"Created starter corpus at {args.corpus}")

  config = ModelConfig(
    vocab_size=settings.vocab_size,
    d_model=settings.d_model,
    n_heads=settings.n_heads,
    n_layers=settings.n_layers,
    d_ff=settings.d_ff,
    max_seq_len=settings.max_seq_len,
  )

  weights_out = Path(settings.model_weights_path)
  tokenizer_out = Path(settings.tokenizer_path)

  print("Training custom model (100% owned, no external models)...")
  train_model(
    corpus_path=args.corpus,
    weights_out=weights_out,
    tokenizer_out=tokenizer_out,
    config=config,
    epochs=args.epochs,
    learning_rate=args.lr,
    seq_len=args.seq_len,
  )
  print(f"Saved weights: {weights_out}")
  print(f"Saved tokenizer: {tokenizer_out}")


if __name__ == "__main__":
  main()
