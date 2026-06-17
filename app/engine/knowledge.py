"""Scalable TF-IDF knowledge retrieval engine.

100% custom and free: pure Python + NumPy, no third-party models and no
pretrained weights. Works across languages (Unicode word tokenization) and
scales to thousands of entries on low-memory free hosting via an inverted index.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Common function words (mostly English) used only to gate coincidental matches.
# We keep them in scoring, but a valid match must share a non-stopword word
# OR be a near-exact phrase match.
_STOPWORDS = {
  "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being",
  "do", "does", "did", "to", "of", "in", "on", "at", "for", "with", "and", "or",
  "but", "if", "then", "else", "i", "you", "he", "she", "it", "we", "they",
  "me", "my", "your", "our", "their", "this", "that", "these", "those", "what",
  "which", "who", "whom", "how", "when", "where", "why", "can", "could", "would",
  "should", "will", "shall", "may", "might", "must", "about", "tell", "give",
  "please", "want", "need", "get", "from", "as", "by", "so", "not", "no", "yes",
  "there", "here", "into", "out", "up", "down", "over", "more", "some", "any",
}


def tokenize(text: str) -> list[str]:
  return _TOKEN_RE.findall(text.lower())


class KnowledgeBase:
  def __init__(self, entries: list[tuple[str, str]]) -> None:
    # entries: list of (match_text, answer)
    self._answers: list[str] = []
    self._doc_weights: list[dict[str, float]] = []
    self._doc_terms: list[set[str]] = []
    self._idf: dict[str, float] = {}
    self._inverted: dict[str, list[tuple[int, float]]] = {}
    self._build(entries)

  @property
  def size(self) -> int:
    return len(self._answers)

  def _build(self, entries: list[tuple[str, str]]) -> None:
    docs_tokens: list[list[str]] = []
    df: dict[str, int] = {}
    for match_text, answer in entries:
      toks = tokenize(match_text)
      if not toks:
        continue
      self._answers.append(answer)
      docs_tokens.append(toks)
      for t in set(toks):
        df[t] = df.get(t, 0) + 1

    n = len(docs_tokens)
    self._idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

    for doc_id, toks in enumerate(docs_tokens):
      tf: dict[str, int] = {}
      for t in toks:
        tf[t] = tf.get(t, 0) + 1
      weights: dict[str, float] = {}
      for t, c in tf.items():
        weights[t] = (1.0 + math.log(c)) * self._idf.get(t, 0.0)
      norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
      for t in weights:
        weights[t] /= norm
      self._doc_weights.append(weights)
      self._doc_terms.append(set(weights.keys()))
      for t, w in weights.items():
        self._inverted.setdefault(t, []).append((doc_id, w))

  def search(self, query: str) -> tuple[str | None, float]:
    if not self._answers:
      return None, 0.0
    q_toks = tokenize(query)
    if not q_toks:
      return None, 0.0
    q_all = set(q_toks)

    q_tf: dict[str, int] = {}
    for t in q_toks:
      q_tf[t] = q_tf.get(t, 0) + 1
    q_weights: dict[str, float] = {}
    for t, c in q_tf.items():
      if t in self._idf:
        q_weights[t] = (1.0 + math.log(c)) * self._idf[t]
    norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
    for t in q_weights:
      q_weights[t] /= norm

    scores: dict[int, float] = {}
    for t, qw in q_weights.items():
      for doc_id, dw in self._inverted.get(t, ()):  # inverted index lookup
        scores[doc_id] = scores.get(doc_id, 0.0) + qw * dw

    if not scores:
      return None, 0.0
    best_id = max(scores, key=scores.get)
    best_score = float(scores[best_id])

    # Gate coincidental matches: require a shared meaningful (non-stopword) word,
    # or a near-exact phrase overlap (handles all-stopword questions).
    # Use ALL query tokens (including ones outside the KB vocabulary) so that
    # unknown content words correctly lower the overlap.
    doc_set = self._doc_terms[best_id]
    shared = q_all & doc_set
    content_shared = shared - _STOPWORDS
    union = q_all | doc_set
    jaccard = len(shared) / len(union) if union else 0.0
    if not content_shared and jaccard < 0.6:
      return None, best_score

    return self._answers[best_id], best_score


def _parse_corpus_pairs(text: str) -> list[tuple[str, str]]:
  from app.engine.tokenizer import ByteTokenizer

  pairs: list[tuple[str, str]] = []
  pending: str | None = None
  for raw in text.splitlines():
    line = raw.strip()
    if not line:
      continue
    if line.startswith("User:"):
      pending = line[len("User:") :].strip()
    elif line.startswith("Assistant:") and pending is not None:
      ans = line[len("Assistant:") :].replace(ByteTokenizer.EOS, "").strip()
      ans = re.sub(r"\s+", " ", ans)
      if ans:
        pairs.append((pending, ans))
      pending = None
  return pairs


def load_knowledge_base(
  knowledge_path: Path | None = None,
  corpus_path: Path | None = None,
) -> KnowledgeBase:
  """Load entries from a JSONL knowledge file and/or the chat corpus."""
  entries: list[tuple[str, str]] = []

  if knowledge_path and knowledge_path.exists():
    for line in knowledge_path.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except json.JSONDecodeError:
        continue
      answer = (obj.get("a") or obj.get("answer") or obj.get("text") or "").strip()
      question = (obj.get("q") or obj.get("question") or obj.get("title") or answer).strip()
      if answer:
        entries.append((question, answer))

  if corpus_path and corpus_path.exists():
    entries.extend(_parse_corpus_pairs(corpus_path.read_text(encoding="utf-8")))

  return KnowledgeBase(entries)
