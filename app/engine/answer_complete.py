"""Detect incomplete answers and drive multi-pass continuation."""

from __future__ import annotations

import re

CONTINUE_PROMPT = (
  "Continue exactly where you stopped. Complete every remaining section, "
  "code block, list item, and key takeaway. Do not repeat content you already wrote. "
  "End with a clear, finished summary."
)


def is_incomplete_answer(text: str) -> bool:
  """Heuristic: answer likely cut off before the model finished."""
  t = (text or "").strip()
  if len(t) < 40:
    return True

  if t.count("```") % 2 == 1:
    return True

  lines = [ln.rstrip() for ln in t.splitlines() if ln.strip()]
  if not lines:
    return True

  last = lines[-1].strip()

  if re.match(r"^#{1,4}\s+\S", last) and len(last) < 120:
    return True

  if last.startswith(("-", "*", "|")) and not re.search(r"[.!?)\]`\"']\s*$", last):
    return True

  if re.match(r"^\d+\.\s", last) and not re.search(r"[.!?)\]`\"']\s*$", last):
    return True

  if len(t) > 180 and not re.search(r"[.!?)\]`\"']\s*$", t):
    return True

  open_sections = len(re.findall(r"^#{2,4}\s+", t, flags=re.MULTILINE))
  if open_sections >= 2 and "takeaway" in t.lower() and "conclusion" not in t.lower():
    tail = t[-400:].lower()
    if "takeaway" in tail and not re.search(r"[.!?]\s*$", t):
      return True

  return False


def merge_continuation(previous: str, chunk: str) -> str:
  prev = (previous or "").strip()
  nxt = (chunk or "").strip()
  if not nxt:
    return prev
  if not prev:
    return nxt
  if nxt.startswith(prev[: min(80, len(prev))]):
    return nxt
  return f"{prev}\n\n{nxt}".strip()
