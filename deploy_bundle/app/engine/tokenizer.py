"""Custom byte-level tokenizer — trained and owned entirely by this project."""

from __future__ import annotations

import json
import re
from pathlib import Path


class ByteTokenizer:
  """Lightweight BPE-style tokenizer built from your own training corpus."""

  PAD = "<|pad|>"
  BOS = "<|bos|>"
  EOS = "<|eos|>"
  UNK = "<|unk|>"
  SPECIALS = (PAD, BOS, EOS, UNK)

  def __init__(self, vocab: dict[str, int] | None = None) -> None:
    self._vocab: dict[str, int] = vocab or {}
    self._inv: dict[int, str] = {}

  @property
  def vocab_size(self) -> int:
    return len(self._vocab)

  def _rebuild_inverse(self) -> None:
    self._inv = {i: t for t, i in self._vocab.items()}

  @classmethod
  def train_from_text(
    cls,
    text: str,
    vocab_size: int = 8192,
    min_freq: int = 2,
  ) -> ByteTokenizer:
    """Train tokenizer on raw text without any external model files."""
    tok = cls()
    words = re.findall(r"\S+|\s+", text)
    freq: dict[str, int] = {}
    for w in words:
      freq[w] = freq.get(w, 0) + 1

    vocab: dict[str, int] = {s: i for i, s in enumerate(cls.SPECIALS)}
    idx = len(vocab)

    for token, count in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
      if count < min_freq:
        continue
      if token not in vocab and idx < vocab_size:
        vocab[token] = idx
        idx += 1

    # Byte fallback for unseen characters
    for b in range(256):
      ch = chr(b)
      if ch not in vocab and idx < vocab_size:
        vocab[ch] = idx
        idx += 1

    tok._vocab = vocab
    tok._rebuild_inverse()
    return tok

  def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
    if not self._vocab:
      raise RuntimeError("Tokenizer is not loaded or trained")

    ids: list[int] = []
    if add_bos:
      ids.append(self._vocab[self.BOS])

    i = 0
    while i < len(text):
      matched = False
      for length in range(min(32, len(text) - i), 0, -1):
        piece = text[i : i + length]
        if piece in self._vocab:
          ids.append(self._vocab[piece])
          i += length
          matched = True
          break
      if not matched:
        ch = text[i]
        ids.append(self._vocab.get(ch, self._vocab[self.UNK]))
        i += 1

    if add_eos:
      ids.append(self._vocab[self.EOS])
    return ids

  def decode(self, ids: list[int], skip_specials: bool = True) -> str:
    if not self._inv:
      self._rebuild_inverse()

    parts: list[str] = []
    special_set = set(self.SPECIALS)
    for token_id in ids:
      piece = self._inv.get(token_id, self.UNK)
      if skip_specials and piece in special_set:
        continue
      parts.append(piece)
    return "".join(parts)

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(self._vocab, ensure_ascii=False, indent=2), encoding="utf-8")

  @classmethod
  def load(cls, path: Path) -> ByteTokenizer:
    data = json.loads(path.read_text(encoding="utf-8"))
    tok = cls(vocab=data)
    tok._rebuild_inverse()
    return tok
