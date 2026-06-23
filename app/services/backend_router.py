"""Resolve chat backend from API model field or user prompt prefix.

Supported backends: custom | ollama | llm | auto

Prompt prefixes (stripped before inference):
  /custom What is gold?
  /ollama Explain Python
  /llm Write an email
  /auto Best available backend
  @ollama hello
"""

from __future__ import annotations

import copy
import re

VALID_BACKENDS = frozenset({"custom", "gemma", "ollama", "llm", "auto"})

_PREFIX_RE = re.compile(
  r"^(?:/)?@?(custom|gemma|ollama|llm|auto)\b[:\s,-]*",
  re.IGNORECASE,
)

_CUSTOM_FAILURE_MARKERS = (
  "could not produce a confident answer through inference",
  "need more training",
  "train_worldwide.py",
  "run scripts/train.py",
)


def normalize_backend(name: str | None, default: str) -> str:
  if not name:
    return default if default in VALID_BACKENDS else "auto"
  low = name.strip().lower()
  if low in VALID_BACKENDS:
    return low
  if low.startswith("custom") or low.startswith("nexus"):
    return "custom"
  if "ollama" in low:
    return "ollama"
  if "gemma" in low:
    return "gemma"
  if low.endswith(".gguf") or low.startswith("llm") or low.startswith("qwen"):
    return "llm"
  return default if default in VALID_BACKENDS else "auto"


def _last_user_message(messages: list[dict[str, str]]) -> tuple[int, str] | tuple[None, None]:
  for i in range(len(messages) - 1, -1, -1):
    if messages[i].get("role") == "user":
      return i, (messages[i].get("content") or "")
  return None, None


def parse_prompt_backend(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
  """Return (backend_from_prefix, messages_with_prefix_stripped)."""
  idx, text = _last_user_message(messages)
  if idx is None:
    return None, messages
  m = _PREFIX_RE.match(text.strip())
  if not m:
    return None, messages
  backend = m.group(1).lower()
  cleaned = _PREFIX_RE.sub("", text.strip(), count=1).strip()
  out = copy.deepcopy(messages)
  out[idx] = {**out[idx], "content": cleaned or text.strip()}
  return backend, out


def resolve_backend(
  messages: list[dict[str, str]],
  *,
  model_field: str | None,
  default_backend: str,
  explicit_backend: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
  """Pick backend: explicit kwarg > API model > prompt prefix > .env default."""
  if explicit_backend:
    return normalize_backend(explicit_backend, "auto"), messages

  from_model = normalize_backend(model_field, "") if model_field else ""
  if from_model in VALID_BACKENDS:
    return from_model, messages

  from_prompt, cleaned = parse_prompt_backend(messages)
  if from_prompt:
    return from_prompt, cleaned

  default = normalize_backend(default_backend, "auto")
  if default_backend.lower().strip() in ("prompt", "multi"):
    return "auto", messages
  return default, messages


def is_custom_failure(text: str) -> bool:
  low = (text or "").lower()
  return any(m in low for m in _CUSTOM_FAILURE_MARKERS)


def is_low_quality_output(text: str) -> bool:
  """Detect failed custom inference (training message or garbage tokens)."""
  if is_custom_failure(text):
    return True
  t = (text or "").strip()
  if len(t) < 30:
    return True
  low = t.lower()
  if "assistant:" in low or low.startswith("user:"):
    return True
  if "knowledge.jsonl" in low or "import_dataset" in low:
    return True
  if t.count("category-aware") >= 1:
    return True
  if t.count("(") >= 4 and t.count(")") >= 4:
    return True
  if re.search(r"\(\w{1,4}\s", t):
    return True
  words = re.findall(r"\w+", t, flags=re.UNICODE)
  if len(words) < 20:
    return True
  short = sum(1 for w in words if len(w) <= 2)
  if short / max(len(words), 1) > 0.35:
    return True
  # Too many non-dictionary tokens
  weird = sum(1 for w in words if not re.match(r"^[a-zA-Z\u00C0-\uFFFF]{2,}$", w))
  if weird / max(len(words), 1) > 0.25:
    return True
  return False
