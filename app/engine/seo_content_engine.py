"""Advanced SEO content engine — worldwide, multilingual, category-aware.

Uses data/seo_content_knowledge.jsonl for training guidance.
100% custom — no GPT, Claude, Gemini.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import re

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEO_CONTENT_KB_PATH = PROJECT_ROOT / "data" / "seo_content_knowledge.jsonl"

_CATEGORIES: dict[str, dict[str, Any]] = {
  "blog_article": {
    "label": "Blog Article",
    "description": "Long-form SEO blog posts and guides",
    "default_tone": "professional",
  },
  "how_to_guide": {
    "label": "How-To Guide",
    "description": "Step-by-step tutorials that rank for informational queries",
    "default_tone": "professional",
  },
  "listicle": {
    "label": "Listicle",
    "description": "Numbered tips, tools, or best-of articles",
    "default_tone": "casual",
  },
  "landing_page": {
    "label": "Landing Page Copy",
    "description": "Conversion-focused service or product landing pages",
    "default_tone": "professional",
  },
  "product_description": {
    "label": "Product Description",
    "description": "E-commerce and SaaS product page copy",
    "default_tone": "professional",
  },
  "local_seo": {
    "label": "Local SEO",
    "description": "City/region-focused pages for local businesses",
    "default_tone": "friendly",
  },
  "news_update": {
    "label": "News / Update",
    "description": "Timely announcements and industry news posts",
    "default_tone": "formal",
  },
  "ecommerce": {
    "label": "E-commerce SEO",
    "description": "Category pages, buying guides, comparison content",
    "default_tone": "professional",
  },
}

# Only these four tones are supported (matches product UI).
_VALID_TONES = ["professional", "casual", "friendly", "formal"]

_TONE_HINTS: dict[str, str] = {
  "professional": "Clear, confident, and business-appropriate. Polished but accessible.",
  "casual": "Relaxed and conversational. Approachable language, easy to read.",
  "friendly": "Warm, helpful, and welcoming. Supportive voice that builds trust.",
  "formal": "Structured and respectful. Suited for corporate, legal, or academic audiences.",
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en", "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
  "german": "de", "de": "de", "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar", "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh", "korean": "ko", "ko": "ko",
  "italian": "it", "it": "it", "russian": "ru", "ru": "ru",
  "bengali": "bn", "bn": "bn", "tamil": "ta", "ta": "ta",
  "marathi": "mr", "mr": "mr", "urdu": "ur", "ur": "ur",
  "vietnamese": "vi", "vi": "vi", "thai": "th", "th": "th",
  "dutch": "nl", "nl": "nl", "polish": "pl", "pl": "pl",
  "turkish": "tr", "tr": "tr", "indonesian": "id", "id": "id",
}

_seo_kb: KnowledgeBase | None = None


def get_seo_kb() -> KnowledgeBase:
  global _seo_kb
  if _seo_kb is None:
    _seo_kb = load_knowledge_base(knowledge_path=SEO_CONTENT_KB_PATH)
  return _seo_kb


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def normalize_tone(tone: str | None, category: str | None = None) -> str:
  if tone:
    t = tone.strip().lower()
    if t in _VALID_TONES:
      return t
  cat = normalize_category(category)
  default = _CATEGORIES[cat]["default_tone"]
  return default if default in _VALID_TONES else "professional"


def tone_hint(tone: str) -> str:
  return _TONE_HINTS.get(tone, _TONE_HINTS["professional"])


def normalize_category(category: str | None) -> str:
  if not category:
    return "blog_article"
  key = category.strip().lower().replace(" ", "_").replace("-", "_")
  if key in _CATEGORIES:
    return key
  aliases = {
    "blog": "blog_article", "article": "blog_article", "guide": "how_to_guide",
    "howto": "how_to_guide", "landing": "landing_page", "product": "product_description",
    "local": "local_seo", "news": "news_update", "ecommerce": "ecommerce",
  }
  return aliases.get(key, "blog_article")


def supported_categories() -> list[dict[str, str]]:
  return [
    {"id": k, "label": v["label"], "description": v["description"], "default_tone": v["default_tone"]}
    for k, v in _CATEGORIES.items()
  ]


def supported_tones() -> list[dict[str, str]]:
  return [
    {"id": t, "label": t.capitalize()}
    for t in _VALID_TONES
  ]


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"}, {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"}, {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"}, {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"}, {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"}, {"name": "Korean", "code": "ko"},
    {"name": "Italian", "code": "it"}, {"name": "Russian", "code": "ru"},
    {"name": "Bengali", "code": "bn"}, {"name": "Tamil", "code": "ta"},
    {"name": "Marathi", "code": "mr"}, {"name": "Urdu", "code": "ur"},
    {"name": "Vietnamese", "code": "vi"}, {"name": "Thai", "code": "th"},
    {"name": "Dutch", "code": "nl"}, {"name": "Polish", "code": "pl"},
    {"name": "Turkish", "code": "tr"}, {"name": "Indonesian", "code": "id"},
  ]


def get_guidance(topic: str, category: str, language: str | None) -> str:
  kb = get_seo_kb()
  lang = bcp47(language)
  queries = [
    f"SEO content {category} best practices",
    f"SEO article writing {topic}",
    f"SEO content multilingual {lang}",
  ]
  chunks: list[str] = []
  for q in queries:
    answer, score = kb.search(q)
    if answer and score > 0.05 and answer not in chunks:
      chunks.append(answer)
  return "\n\n".join(chunks[:2])


def category_structure_hint(category: str) -> str:
  hints = {
    "blog_article": "Use intro, 3-5 H2 sections, bullet lists where helpful, and a conclusion with CTA.",
    "how_to_guide": "Use numbered steps under H2/H3, prerequisites section, and FAQ-style tips.",
    "listicle": "Use H2 for each list item (e.g. '## 1. First tip'), short paragraphs, summary table optional.",
    "landing_page": "Use hero value prop, benefits bullets, social proof section, and strong CTA.",
    "product_description": "Use features, benefits, use cases, specs, and trust signals.",
    "local_seo": "Mention location naturally, local benefits, service area, and contact CTA.",
    "news_update": "Lead with the key update, context, impact, and what readers should do next.",
    "ecommerce": "Use buying guide structure, comparison points, pros/cons, and recommendation.",
  }
  return hints.get(category, hints["blog_article"])


def build_outline(topic: str, keywords: list[str], category: str) -> list[str]:
  """Article section outline (H2-level)."""
  primary = (keywords[0] if keywords else topic).strip().title()
  if category == "how_to_guide":
    return [
      f"What You Need Before Starting With {primary}",
      f"Step 1: Understand {primary} Fundamentals",
      "Step 2: Apply Proven Techniques",
      "Step 3: Optimize and Scale Results",
      "Common Mistakes to Avoid",
      "Conclusion and Next Steps",
    ]
  if category == "listicle":
    return [
      f"1. Define Your {primary} Strategy",
      f"2. Choose the Right Tools for {primary}",
      "3. Create High-Quality Original Content",
      "4. Measure and Improve Continuously",
      "5. Stay Updated With Industry Trends",
      "Summary and Action Plan",
    ]
  if category == "landing_page":
    return [
      "Hero Value Proposition",
      "Key Benefits",
      "How It Works",
      "Social Proof and Trust Signals",
      "Call to Action",
    ]
  return [
    f"Introduction to {primary}",
    f"Why {primary} Matters Today",
    f"Key Benefits of {primary}",
    "Best Practices for Worldwide Audiences",
    f"How to Get Started With {primary}",
    "Conclusion",
  ]


def build_faqs(topic: str, keywords: list[str], *, language: str | None = None) -> list[dict[str, str]]:
  """FAQ list — question + answer pairs."""
  primary = (keywords[0] if keywords else topic).strip()
  lang_note = f" ({language})" if language else ""
  return [
    {
      "question": f"What is {primary}?",
      "answer": (
        f"{primary.title()} is a proven approach used by professionals worldwide{lang_note} "
        "to improve visibility, engagement, and measurable results."
      ),
    },
    {
      "question": f"How long does {primary} take to show results?",
      "answer": (
        "Most strategies show meaningful progress within 8–12 weeks when applied consistently "
        "with quality content and proper optimization."
      ),
    },
    {
      "question": f"Who should focus on {primary}?",
      "answer": (
        "Marketers, business owners, creators, and teams who want sustainable growth "
        "across search, social, and content channels globally."
      ),
    },
    {
      "question": f"What are the best practices for {primary}?",
      "answer": (
        "Focus on user intent, original research, clear structure, mobile-friendly pages, "
        "and regular publishing — avoid keyword stuffing and thin content."
      ),
    },
  ]


def extract_outline_from_body(body: str) -> list[str]:
  outline: list[str] = []
  for line in (body or "").split("\n"):
    m = re.match(r"^##\s+(.+)$", line.strip())
    if m:
      title = re.sub(r"[*_`]", "", m.group(1)).strip()
      if title.lower() not in ("frequently asked questions", "faq", "faqs"):
        outline.append(title)
  return outline


def extract_faqs_from_body(body: str) -> list[dict[str, str]]:
  faqs: list[dict[str, str]] = []
  in_faq = False
  current_q = ""
  for line in (body or "").split("\n"):
    stripped = line.strip()
    if re.match(r"^##\s+(frequently asked questions|faqs?)\s*$", stripped, re.I):
      in_faq = True
      continue
    if not in_faq:
      continue
    if stripped.startswith("### "):
      if current_q:
        faqs.append({"question": current_q, "answer": ""})
      current_q = stripped[4:].strip()
    elif stripped and current_q:
      if faqs and faqs[-1]["question"] == current_q and not faqs[-1]["answer"]:
        faqs[-1]["answer"] = stripped
      elif faqs and faqs[-1]["question"] == current_q:
        faqs[-1]["answer"] += " " + stripped
      else:
        faqs.append({"question": current_q, "answer": stripped})
  if current_q and (not faqs or faqs[-1]["question"] != current_q):
    faqs.append({"question": current_q, "answer": ""})
  return [f for f in faqs if f.get("question")]


def strip_faq_section(body: str) -> str:
  """Remove FAQ block from article body when FAQs are returned separately."""
  lines = (body or "").split("\n")
  out: list[str] = []
  skip = False
  for line in lines:
    if re.match(r"^##\s+(frequently asked questions|faqs?)\s*$", line.strip(), re.I):
      skip = True
      continue
    if skip and line.strip().startswith("## "):
      skip = False
    if not skip:
      out.append(line)
  return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def quality_report(title: str, meta: str, content: str, keywords: list[str]) -> dict[str, Any]:
  issues: list[str] = []
  score = 100
  if len(title) < 20:
    issues.append("title_short"); score -= 10
  if len(meta) < 50:
    issues.append("meta_short"); score -= 10
  if len(meta) > 160:
    issues.append("meta_long"); score -= 5
  wc = len(content.split())
  if wc < 120:
    issues.append("content_short"); score -= 20
  if not content.count("##"):
    issues.append("missing_h2"); score -= 15
  if keywords:
    primary = keywords[0].lower()
    if primary not in (title + content).lower():
      issues.append("primary_keyword_missing"); score -= 15
  return {
    "seo_score": max(0, min(100, score)),
    "seo_ready": score >= 70,
    "issues": issues,
    "heading_count": content.count("##"),
    "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
  }
