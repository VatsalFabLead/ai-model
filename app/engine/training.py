"""Training utilities for the custom transformer (NumPy only)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from app.engine.tokenizer import ByteTokenizer
from app.engine.transformer import CustomTransformer, ModelConfig


def _cross_entropy(logits: np.ndarray, target_id: int) -> tuple[float, np.ndarray]:
  shifted = logits - np.max(logits)
  exp = np.exp(shifted)
  probs = exp / np.sum(exp)
  loss = -math.log(probs[target_id] + 1e-9)
  grad = probs.copy()
  grad[target_id] -= 1.0
  return loss, grad


def _clip_gradients(weights: dict[str, np.ndarray], max_norm: float = 1.0) -> None:
  total = 0.0
  for arr in weights.values():
    total += float(np.sum(arr**2))
  norm = math.sqrt(total)
  if norm > max_norm:
    scale = max_norm / (norm + 1e-6)
    for key in weights:
      weights[key] *= scale


def build_sequences(text: str, tokenizer: ByteTokenizer, seq_len: int) -> list[np.ndarray]:
  ids = tokenizer.encode(text, add_bos=True, add_eos=True)
  sequences: list[np.ndarray] = []
  for start in range(0, len(ids) - seq_len, max(1, seq_len // 2)):
    chunk = ids[start : start + seq_len + 1]
    if len(chunk) == seq_len + 1:
      sequences.append(np.array(chunk, dtype=np.int32))
  return sequences


def train_model(
  corpus_path: Path,
  weights_out: Path,
  tokenizer_out: Path,
  config: ModelConfig,
  epochs: int = 3,
  learning_rate: float = 3e-4,
  seq_len: int = 64,
) -> CustomTransformer:
  """Train custom model on local corpus — no external weights."""
  text = corpus_path.read_text(encoding="utf-8")
  tokenizer = ByteTokenizer.train_from_text(text, vocab_size=config.vocab_size)
  tokenizer.save(tokenizer_out)

  model = CustomTransformer(config)
  sequences = build_sequences(text, tokenizer, seq_len)
  if not sequences:
    raise ValueError("Corpus too small for training. Add more text to data/corpus.txt")

  for epoch in range(epochs):
    total_loss = 0.0
    rng = np.random.default_rng(epoch)
    order = rng.permutation(len(sequences))

    for idx in order:
      tokens = sequences[idx]
      input_ids = tokens[:-1]
      target_id = int(tokens[-1])

      logits = model.forward(input_ids.reshape(1, -1))[0, -1]
      loss, grad_logits = _cross_entropy(logits, target_id)
      total_loss += loss

      # Lightweight update: nudge lm_head and token embeddings toward target
      last_h = model.weights["token_emb"][input_ids[-1]] + model.weights["pos_emb"][len(input_ids) - 1]
      model.weights["lm_head"] -= learning_rate * np.outer(last_h, grad_logits)
      model.weights["token_emb"][input_ids[-1]] -= learning_rate * (grad_logits @ model.weights["lm_head"].T)

    avg = total_loss / max(len(sequences), 1)
    print(f"Epoch {epoch + 1}/{epochs} — loss: {avg:.4f}")

  model.save(weights_out)
  return model
