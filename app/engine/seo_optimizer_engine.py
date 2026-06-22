"""SEO Optimizer engine — analysis metrics, categories, multilingual training.

100% custom — no GPT, Claude, Gemini.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEO_OPTIMIZER_KB_PATH = PROJECT_ROOT / "data" / "seo_optimizer_knowledge.jsonl"

_VALID_TONES = ["professional", "casual", "friendly", "formal"]

_TONE_HINTS: dict[str, str] = {
  "professional": "Clear, confident, business-appropriate.",
  "casual": "Relaxed, conversational, approachable.",
  "friendly": "Warm, helpful, welcoming.",
  "formal": "Structured, respectful, corporate or academic.",
}

_CATEGORIES: dict[str, dict[str, str]] = {
  "blog_article": {"label": "Blog Article", "default_tone": "professional"},
  "landing_page": {"label": "Landing Page", "default_tone": "professional"},
  "product_description": {"label": "Product Description", "default_tone": "professional"},
  "email_copy": {"label": "Email Copy", "default_tone": "friendly"},
  "social_post": {"label": "Social Post", "default_tone": "casual"},
  "local_seo": {"label": "Local SEO Page", "default_tone": "friendly"},
  "technical_doc": {"label": "Technical Documentation", "default_tone": "formal"},
  "ecommerce": {"label": "E-commerce Copy", "default_tone": "professional"},
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en", "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
  "german": "de", "de": "de", "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar", "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh",
}

_seo_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
  global _seo_kb
  if _seo_kb is None:
    _seo_kb = load_knowledge_base(knowledge_path=SEO_OPTIMIZER_KB_PATH)
  return _seo_kb


def reload_kb() -> KnowledgeBase:
  """Reload training data after import (no server restart needed in dev)."""
  global _seo_kb
  _seo_kb = load_knowledge_base(knowledge_path=SEO_OPTIMIZER_KB_PATH)
  return _seo_kb


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def normalize_tone(tone: str | None, category: str | None = None) -> str:
  if tone and tone.strip().lower() in _VALID_TONES:
    return tone.strip().lower()
  cat = normalize_category(category)
  default = _CATEGORIES.get(cat, {}).get("default_tone", "professional")
  return default if default in _VALID_TONES else "professional"


def normalize_category(category: str | None) -> str:
  if not category:
    return "blog_article"
  key = category.strip().lower().replace(" ", "_").replace("-", "_")
  return key if key in _CATEGORIES else "blog_article"


def tone_hint(tone: str) -> str:
  return _TONE_HINTS.get(tone, _TONE_HINTS["professional"])


def supported_categories() -> list[dict[str, str]]:
  return [{"id": k, **v} for k, v in _CATEGORIES.items()]


def supported_tones() -> list[dict[str, str]]:
  return [{"id": t, "label": t.capitalize()} for t in _VALID_TONES]


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"}, {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"}, {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"}, {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"}, {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"},
  ]


def _syllables(word: str) -> int:
  w = word.lower().strip(".,!?;:'\"")
  if len(w) <= 2:
    return 1
  vowels = "aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ"
  count = 0
  prev_v = False
  for ch in w:
    is_v = ch in vowels
    if is_v and not prev_v:
      count += 1
    prev_v = is_v
  return max(1, count)


def count_sentences(text: str) -> int:
  parts = re.split(r"[.!?]+", text or "")
  return max(1, len([p for p in parts if p.strip()]))


def count_words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE))


def count_characters(text: str) -> int:
  return len((text or "").strip())


def readability_score(text: str) -> float:
  """Simplified Flesch Reading Ease (0–100, higher = easier)."""
  words = re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE)
  if not words:
    return 0.0
  sentences = count_sentences(text)
  syllable_count = sum(_syllables(w) for w in words)
  asl = len(words) / sentences
  asw = syllable_count / len(words)
  score = 206.835 - (1.015 * asl) - (84.6 * asw)
  return round(max(0.0, min(100.0, score)), 2)


def content_metrics(text: str) -> dict[str, Any]:
  return {
    "readability_score": readability_score(text),
    "word_count": count_words(text),
    "character_count": count_characters(text),
    "sentence_count": count_sentences(text),
  }


def analyze_issues(content: str, keywords: list[str] | None = None) -> list[dict[str, str]]:
  issues: list[dict[str, str]] = []
  text = content or ""
  wc = count_words(text)

  if wc < 50:
    issues.append({"type": "length", "priority": "high", "message": "Content is too short for strong SEO (aim for 300+ words for articles)."})
  if "##" not in text and wc > 150:
    issues.append({"type": "structure", "priority": "high", "message": "Add H2 (##) subheadings to improve scanability and rankings."})
  if not re.search(r"^#\s+", text, re.MULTILINE) and wc > 100:
    issues.append({"type": "structure", "priority": "medium", "message": "Consider adding a clear H1 title at the top."})

  long_paras = [p for p in re.split(r"\n\s*\n", text) if count_words(p) > 120 and not p.strip().startswith("#")]
  if long_paras:
    issues.append({"type": "readability", "priority": "medium", "message": f"{len(long_paras)} paragraph(s) are very long — split into shorter blocks."})

  long_sents = [s for s in re.split(r"[.!?]+", text) if count_words(s) > 30]
  if len(long_sents) >= 2:
    issues.append({"type": "readability", "priority": "medium", "message": "Some sentences are too long — aim for 15–20 words per sentence."})

  if keywords:
    primary = keywords[0].lower()
    if primary not in text.lower()[:400]:
      issues.append({"type": "keyword", "priority": "high", "message": f"Primary keyword '{keywords[0]}' should appear in the first paragraph."})
    density = text.lower().count(primary) / max(wc, 1) * 100
    if density > 3.5:
      issues.append({"type": "keyword", "priority": "high", "message": "Keyword density may be too high — reduce stuffing for natural flow."})
    elif density < 0.3 and wc > 100:
      issues.append({"type": "keyword", "priority": "medium", "message": f"Use primary keyword '{keywords[0]}' a few more times naturally."})

  if not re.search(r"(conclusion|summary|in summary|to sum up|finally)", text, re.IGNORECASE) and wc > 250:
    issues.append({"type": "structure", "priority": "low", "message": "Add a conclusion section with a clear takeaway or CTA."})

  return issues


def seo_score_from_analysis(metrics: dict[str, Any], issues: list[dict[str, str]]) -> int:
  score = 100
  for issue in issues:
    p = issue.get("priority", "low")
    score -= {"high": 18, "medium": 10, "low": 5}.get(p, 5)
  if metrics.get("readability_score", 0) < 40:
    score -= 15
  elif metrics.get("readability_score", 0) < 55:
    score -= 8
  if metrics.get("word_count", 0) < 100:
    score -= 10
  return max(0, min(100, score))
