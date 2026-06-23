"""Knowledge retriever using the trained model's own embeddings.

100% custom and free: no third-party models. Embeddings come from the custom
transformer's learned token embedding table. Used as a high-precision layer on
top of neural generation so known questions return clean answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.engine.tokenizer import ByteTokenizer
from app.engine.transformer import CustomTransformer


def parse_corpus(text: str) -> list[tuple[str, str]]:
  """Extract (question, answer) pairs from the chat-formatted corpus."""
  pairs: list[tuple[str, str]] = []
  pending_user: str | None = None
  answer_lines: list[str] = []

  def flush() -> None:
    nonlocal pending_user, answer_lines
    if pending_user is None or not answer_lines:
      pending_user = None
      answer_lines = []
      return
    answer = "\n".join(answer_lines)
    answer = answer.replace(ByteTokenizer.EOS, "").strip()
    if answer:
      pairs.append((pending_user, answer))
    pending_user = None
    answer_lines = []

  for raw in text.splitlines():
    line = raw.rstrip()
    stripped = line.strip()
    if not stripped:
      if answer_lines:
        answer_lines.append("")
      continue
    if stripped.startswith("User:"):
      flush()
      pending_user = stripped[len("User:") :].strip()
      continue
    if stripped.startswith("Assistant:") and pending_user is not None:
      answer_lines = [stripped[len("Assistant:") :].strip()]
      continue
    if pending_user is not None and answer_lines:
      answer_lines.append(line)
      continue
    flush()
  flush()
  return pairs


class KnowledgeRetriever:
  def __init__(
    self,
    model: CustomTransformer,
    tokenizer: ByteTokenizer,
    pairs: list[tuple[str, str]],
  ) -> None:
    self._model = model
    self._tokenizer = tokenizer
    self._pairs = pairs
    self._matrix = self._build_matrix()

  def _embed(self, text: str) -> np.ndarray:
    ids = self._tokenizer.encode(text.lower(), add_bos=False, add_eos=False)
    table = self._model.weights["token_emb"]
    if not ids:
      return np.zeros(table.shape[1], dtype=np.float32)
    vec = table[ids].mean(axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

  def _build_matrix(self) -> np.ndarray:
    if not self._pairs:
      return np.zeros((0, self._model.weights["token_emb"].shape[1]), dtype=np.float32)
    return np.stack([self._embed(q) for q, _ in self._pairs])

  def retrieve(self, query: str) -> tuple[str | None, float]:
    if not self._pairs:
      return None, 0.0
    q_vec = self._embed(query)
    sims = self._matrix @ q_vec
    best = int(np.argmax(sims))
    return self._pairs[best][1], float(sims[best])


def load_retriever(
  model: CustomTransformer,
  tokenizer: ByteTokenizer,
  corpus_path: Path,
) -> KnowledgeRetriever:
  text = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""
  pairs = parse_corpus(text)
  return KnowledgeRetriever(model, tokenizer, pairs)
