"""Answer formatting — optional word limits and source-stripping for chat responses."""

from __future__ import annotations

import re

_SOURCE_PATTERNS = (
  re.compile(r"\n*[_(]*\s*Source:\s*Wikipedia[^\n)]*[_)]*\s*", re.IGNORECASE),
  re.compile(r"\n*[_(]*\s*Source:\s*[^\n)]+\s*[_)]*\s*", re.IGNORECASE),
  re.compile(r"\n*\(Source:[^)]+\)\s*", re.IGNORECASE),
  re.compile(r"\n*_\(Source:[^)]+\)_\s*", re.IGNORECASE),
)


def count_words(text: str) -> int:
  return len(re.findall(r"\w+", text or "", flags=re.UNICODE))


def strip_source_attribution(text: str) -> str:
  out = (text or "").strip()
  for pat in _SOURCE_PATTERNS:
    out = pat.sub("\n", out)
  return re.sub(r"\n{3,}", "\n\n", out).strip()


def clamp_words(text: str, *, min_words: int, max_words: int) -> str:
  """Trim to max_words. min_words is a target when enough material exists (no padding)."""
  cleaned = strip_source_attribution(text)
  if not cleaned:
    return cleaned
  words = re.findall(r"\S+", cleaned, flags=re.UNICODE)
  if len(words) > max_words:
    trimmed = " ".join(words[:max_words])
    if not trimmed.endswith((".", "!", "?", ":")):
      trimmed += "..."
    return trimmed
  return cleaned


def format_answer_text(text: str, *, min_words: int = 0, max_words: int = 0) -> str:
  """Strip sources; apply word caps only when max_words > 0."""
  cleaned = strip_source_attribution(text)
  if not cleaned or max_words <= 0:
    return cleaned
  return clamp_words(cleaned, min_words=max(0, min_words), max_words=max_words)
