#!/usr/bin/env python3
"""Build structured SEO content training data (custom model only — no GPT/Claude/Gemini).

Adds worldwide, multilingual, category-aware knowledge entries with the structured
output shape: metadata, keywords, outline, content.article, content.tone, faqs.

Usage:
  python scripts/import_seo_content_training.py
  python scripts/import_seo_content_training.py --dry-run
  python scripts/import_seo_content_training.py --epochs 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEO_KB = PROJECT_ROOT / "data" / "seo_content_knowledge.jsonl"

CATEGORIES = [
  "blog_article", "how_to_guide", "listicle", "landing_page",
  "product_description", "local_seo", "news_update", "ecommerce",
]

TONES = ["professional", "casual", "friendly", "formal"]

LANGUAGES = {
  "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
  "de": "German", "pt": "Portuguese", "ar": "Arabic", "ja": "Japanese",
}

STRUCTURED_RULE = (
  "Structured SEO output JSON shape: "
  '{"metadata":{"title":"...","meta_description":"..."},'
  '"keywords":["..."],'
  '"outline":["H2 section 1","H2 section 2"],'
  '"content":{"article":"markdown without FAQ block","tone":"professional"},'
  '"faqs":[{"question":"...","answer":"..."}]}'
)


def _structured_example(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  tone: str,
  language: str,
) -> dict:
  primary = keywords[0]
  return {
    "metadata": {
      "title": f"{primary.title()}: Complete Guide for Global Audiences",
      "meta_description": (
        f"Learn {primary} with proven strategies for {language} readers worldwide. "
        "Actionable tips, expert structure, and FAQs."
      )[:160],
    },
    "keywords": keywords,
    "outline": [
      f"Introduction to {primary.title()}",
      f"Why {primary.title()} Matters",
      "Best Practices Worldwide",
      "Getting Started",
      "Conclusion",
    ],
    "content": {
      "article": (
        f"## Introduction to {primary.title()}\n\n"
        f"This {category.replace('_', ' ')} covers **{primary}** for {language} audiences "
        f"in a {tone} tone.\n\n"
        f"## Why {primary.title()} Matters\n\n"
        "Worldwide teams use structured content to rank, engage, and convert.\n\n"
        "## Conclusion\n\n"
        f"Apply these {primary} steps consistently for sustainable growth."
      ),
      "tone": tone,
    },
    "faqs": [
      {"question": f"What is {primary}?", "answer": f"{primary.title()} is a core strategy for visibility and growth."},
      {"question": f"How long until {primary} shows results?", "answer": "Most campaigns see progress in 8–12 weeks."},
    ],
  }


def build_entries() -> list[dict[str, str]]:
  entries: list[dict[str, str]] = []
  entries.append({
    "q": "SEO content structured output schema",
    "a": STRUCTURED_RULE + " Keep FAQs separate from article body. Outline lists H2 sections only.",
  })
  entries.append({
    "q": "SEO content aesthetic worldwide writing",
    "a": (
      "Aesthetic worldwide SEO: scannable H2/H3, short paragraphs, inclusive language, "
      "mobile-first layout, benefit-led headings, no keyword stuffing, cultural sensitivity, "
      "local examples without stereotypes. Works for all categories and languages."
    ),
  })
  for cat in CATEGORIES:
    entries.append({
      "q": f"SEO content structured {cat} output",
      "a": (
        f"Category {cat}: {STRUCTURED_RULE} "
        f"Use category-specific outline sections. Article markdown uses ## headings matching outline."
      ),
    })
  for tone in TONES:
    entries.append({
      "q": f"SEO content tone {tone} structured",
      "a": f"Set content.tone to '{tone}'. Match vocabulary and sentence rhythm to {tone} style worldwide.",
    })
  for code, name in LANGUAGES.items():
    entries.append({
      "q": f"SEO content multilingual {code} structured",
      "a": (
        f"Write metadata, outline, article, and FAQs in {name}. "
        f"Language code {code}. {STRUCTURED_RULE}"
      ),
    })
  topics = [
    ("email marketing", ["email marketing", "newsletters", "conversions"]),
    ("Flutter app development", ["Flutter", "mobile apps", "cross-platform"]),
    ("local SEO Surat", ["local SEO", "Surat", "Google Business Profile"]),
    ("ecommerce product pages", ["product SEO", "ecommerce", "buying guide"]),
  ]
  for topic, kws in topics:
    for tone in TONES[:2]:
      ex = _structured_example(topic, kws, category="blog_article", tone=tone, language="English")
      entries.append({
        "q": f"SEO content generate {topic} {tone}",
        "a": json.dumps(ex, ensure_ascii=False),
      })
  return entries


def merge_jsonl(path: Path, new_rows: list[dict[str, str]]) -> int:
  existing: dict[str, str] = {}
  if path.exists():
    for line in path.read_text(encoding="utf-8").splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
        q = str(obj.get("q", "")).strip()
        if q:
          existing[q] = line
      except json.JSONDecodeError:
        continue
  added = 0
  for row in new_rows:
    q = row["q"]
    if q not in existing:
      existing[q] = json.dumps(row, ensure_ascii=False)
      added += 1
    else:
      existing[q] = json.dumps(row, ensure_ascii=False)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text("\n".join(existing.values()) + "\n", encoding="utf-8")
  return added


def main() -> int:
  parser = argparse.ArgumentParser(description="Import structured SEO content training data")
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--epochs", type=int, default=0, help="Run train.py after import")
  args = parser.parse_args()

  rows = build_entries()
  print(f"Built {len(rows)} SEO content knowledge entries")
  if args.dry_run:
    for r in rows[:5]:
      print(json.dumps(r, ensure_ascii=False)[:200], "...")
    return 0

  added = merge_jsonl(SEO_KB, rows)
  print(f"Merged into {SEO_KB} ({added} new keys)")

  if args.epochs > 0:
    train = PROJECT_ROOT / "scripts" / "train.py"
    if train.exists():
      cmd = [sys.executable, str(train), "--epochs", str(args.epochs), "--kb", str(SEO_KB)]
      print("Running:", " ".join(cmd))
      subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
