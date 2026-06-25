"""Title & Meta Description engine — worldwide, multilingual, category-aware.

Uses data/title_meta_knowledge.jsonl. 100% custom — no GPT/Claude/Gemini.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TITLE_META_KB_PATH = PROJECT_ROOT / "data" / "title_meta_knowledge.jsonl"

TITLE_MAX = 60
META_MIN = 140
META_MAX = 160

_VALID_TONES = ["professional", "casual", "friendly", "formal"]

_TONE_HINTS: dict[str, str] = {
  "professional": "Clear, credible, business-appropriate CTR copy.",
  "casual": "Relaxed, conversational hooks that feel human.",
  "friendly": "Warm, inviting, trust-building language.",
  "formal": "Polished, authoritative, corporate or academic style.",
}

_CATEGORIES: dict[str, dict[str, str]] = {
  "blog_article": {"label": "Blog Article", "default_tone": "professional"},
  "product_page": {"label": "Product Page", "default_tone": "professional"},
  "landing_page": {"label": "Landing Page", "default_tone": "professional"},
  "local_business": {"label": "Local Business", "default_tone": "friendly"},
  "ecommerce": {"label": "E-commerce", "default_tone": "professional"},
  "saas": {"label": "SaaS / Software", "default_tone": "professional"},
  "news": {"label": "News / Update", "default_tone": "formal"},
  "how_to": {"label": "How-To Guide", "default_tone": "professional"},
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en", "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
  "german": "de", "de": "de", "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar", "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh", "korean": "ko", "ko": "ko",
  "italian": "it", "it": "it", "russian": "ru", "ru": "ru",
}

_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
  global _kb
  if _kb is None:
    _kb = load_knowledge_base(knowledge_path=TITLE_META_KB_PATH)
  return _kb


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
    {"name": "Chinese", "code": "zh"}, {"name": "Korean", "code": "ko"},
  ]


def quality_variation(title: str, meta: str, topic: str) -> dict[str, Any]:
  from app.engine.title_meta_enrichment import score_metadata_pair

  intent = {"primary": "informational", "ctr_pattern": "guide"}
  scores = score_metadata_pair(title, meta, topic, intent, 0)
  return {
    "quality_score": scores["overall_score"],
    "seo_ready": scores["seo_ready"],
    "issues": scores["issues"],
    "seo_score": scores["seo_score"],
    "ctr_score": scores["ctr_score"],
    "overall_score": scores["overall_score"],
  }
