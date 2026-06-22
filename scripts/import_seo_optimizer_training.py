#!/usr/bin/env python3
"""Build and import SEO optimizer training (100% custom — no GPT/Claude/Gemini).

Generates worldwide, multilingual, category/tone-aware knowledge entries and
optional chat corpus pairs for training the custom NumPy transformer.

Usage:
  python scripts/import_seo_optimizer_training.py
  python scripts/import_seo_optimizer_training.py --build-corpus
  python scripts/import_seo_optimizer_training.py --dry-run
  python scripts/import_seo_optimizer_training.py --epochs 8   # also run train.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEO_KB = PROJECT_ROOT / "data" / "seo_optimizer_knowledge.jsonl"
MAIN_KB = PROJECT_ROOT / "data" / "knowledge.jsonl"
SEO_CORPUS = PROJECT_ROOT / "data" / "seo_optimizer_corpus.txt"
MAIN_CORPUS = PROJECT_ROOT / "data" / "corpus.txt"

CATEGORIES = {
  "blog_article": "Blog Article",
  "landing_page": "Landing Page",
  "product_description": "Product Description",
  "email_copy": "Email Copy",
  "social_post": "Social Post",
  "local_seo": "Local SEO Page",
  "technical_doc": "Technical Documentation",
  "ecommerce": "E-commerce Copy",
}

TONES = ["professional", "casual", "friendly", "formal"]

LANGUAGES = {
  "en": "English",
  "hi": "Hindi",
  "es": "Spanish",
  "fr": "French",
  "de": "German",
  "pt": "Portuguese",
  "ar": "Arabic",
  "ja": "Japanese",
  "zh": "Chinese",
}

AUDIENCES = [
  "B2B SaaS buyers worldwide",
  "B2C ecommerce shoppers",
  "India tier-1 and tier-2 city audiences",
  "LATAM Spanish-speaking markets",
  "MENA Arabic-speaking readers",
  "European multilingual readers",
  "Asia-Pacific mobile-first users",
  "Gen Z social-native readers",
  "senior professional decision-makers",
  "developers and technical teams",
  "healthcare YMYL readers",
  "finance and compliance-aware readers",
  "travel and hospitality seekers",
  "education and e-learning students",
  "nonprofit and community audiences",
]

AESTHETIC_RULES = (
  "Aesthetic worldwide copy: short paragraphs (2-4 sentences), generous white space, "
  "scannable H2/H3 hierarchy, active voice, inclusive language, mobile-first line length, "
  "no walls of text, transition words between sections, benefit-led headings, "
  "consistent tone and punctuation, accessible plain language for global audiences."
)

ADVANCED_RULES = (
  "Advanced SEO editing: match search intent, strengthen E-E-A-T signals, semantic keyword "
  "variations, entity clarity, featured-snippet blocks (40-55 words after question H2), "
  "People Also Ask coverage, remove duplicate phrases, natural keyword placement, "
  "conclusion with CTA, internal link placeholders, image alt-text hints where relevant."
)


def _example_optimized(keyword: str, tone: str, category: str) -> str:
  title = keyword.title()
  return (
    f"OPTIMIZED:\n"
    f"# {title}: A Complete Guide for Global Audiences\n\n"
    f"Discover proven strategies for **{keyword}** that work for readers worldwide. "
    f"This {category.replace('_', ' ')} uses a {tone} voice with clear structure and "
    f"natural keyword placement.\n\n"
    f"## Why {title} Matters\n\n"
    f"Teams across industries rely on {keyword} to reach customers, build trust, and "
    f"drive measurable growth. Strong copy balances clarity, scanability, and intent.\n\n"
    f"## Key Strategies\n\n"
    f"- Lead with benefits and outcomes, not jargon\n"
    f"- Use short sentences (15-20 words) and active voice\n"
    f"- Place **{keyword}** naturally in headings and the opening\n"
    f"- Add a clear conclusion with one next step\n\n"
    f"## Conclusion\n\n"
    f"Apply these {tone} improvements to make your {keyword} content easier to read, "
    f"more engaging, and better aligned with search intent worldwide.\n\n"
    f"SUGGESTIONS:\n"
    f"- Added H1 and H2 structure for scanability\n"
    f"- Improved readability with shorter sentences and bullets\n"
    f"- Wove primary keyword '{keyword}' naturally into title and intro\n"
    f"- Applied {tone} tone consistently for a {category.replace('_', ' ')}"
  )


def generate_entries() -> list[dict[str, str]]:
  entries: list[dict[str, str]] = []
  seen: set[str] = set()

  def add(q: str, a: str) -> None:
    key = q.strip().lower()
    if key and key not in seen:
      seen.add(key)
      entries.append({"q": q.strip(), "a": a.strip()})

  add("SEO content optimization best practices readability", (
    "SEO optimization: improve readability (short sentences 15-20 words, short paragraphs "
    "2-4 sentences), add H1 title and H2 subheadings, place primary keyword in first 100 "
    "words, use bullet lists, active voice, transition words, conclusion with CTA. "
    "Target readability score 60-80 for general audiences worldwide."
  ))
  add("SEO optimizer aesthetic worldwide human copy", AESTHETIC_RULES)
  add("SEO optimizer advanced worldwide editing", ADVANCED_RULES)
  add("SEO optimizer inclusive accessible global writing", (
    "Inclusive global copy: plain language, avoid idioms that do not translate culturally, "
    "respect regional spelling, gender-neutral job titles where possible, readable fonts "
    "and structure for screen readers, define acronyms on first use, avoid exclusionary tone."
  ))

  for cat_id, cat_label in CATEGORIES.items():
    add(f"SEO optimizer {cat_id} category best practices", (
      f"{cat_label} optimization: use category-appropriate structure, scannable headings, "
      f"primary keyword in title and intro, {AESTHETIC_RULES} {ADVANCED_RULES}"
    ))
    for tone in TONES:
      add(f"SEO optimizer {cat_id} {tone} tone rewrite", (
        f"Rewrite {cat_label} in a {tone} tone. {tone.capitalize()} voice throughout. "
        f"Preserve facts, improve flow, headings, keyword use, and worldwide readability."
      ))

  for code, name in LANGUAGES.items():
    add(f"SEO optimizer multilingual {name} {code}", (
      f"Optimize content in {name} ({code}): write entirely in {name}, adapt idioms not "
      f"literal translation, culturally relevant examples, Unicode headings, meta-worthy "
      f"intro, keyword in natural {name} form for local search intent worldwide."
    ))

  for audience in AUDIENCES:
    add(f"SEO optimizer audience {audience}", (
      f"Audience: {audience}. Match vocabulary, examples, and CTA to this reader. "
      f"Keep tone respectful and region-appropriate. {AESTHETIC_RULES}"
    ))

  for tone in TONES:
    add(f"SEO optimizer {tone} tone guidelines worldwide", (
      f"{tone.capitalize()} tone worldwide: consistent voice, remove filler, improve "
      f"engagement, preserve meaning, natural keywords. {AESTHETIC_RULES}"
    ))

  topics = [
    "email marketing", "digital marketing", "SEO strategy", "content marketing",
    "social media growth", "ecommerce conversion", "local business SEO",
    "mobile app development", "cloud computing", "cybersecurity basics",
  ]
  for topic in topics:
    for cat_id in ("blog_article", "landing_page", "ecommerce"):
      for tone in ("professional", "friendly"):
        add(
          f"SEO optimizer rewrite {cat_id} {tone} {topic}",
          _example_optimized(topic, tone, cat_id),
        )

  return entries


def build_corpus_pairs(entries: list[dict[str, str]]) -> str:
  lines: list[str] = []
  for row in entries:
    q = row["q"]
    a = row["a"]
    if "OPTIMIZED:" not in a:
      continue
    lines.append(f"User: {q}\nAssistant: {a} <|eos|>\n")
  return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict]:
  rows: list[dict] = []
  if not path.exists():
    return rows
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      continue
  return rows


def main() -> None:
  parser = argparse.ArgumentParser(description="Import SEO optimizer training (custom model only)")
  parser.add_argument("--dry-run", action="store_true", help="Show counts only")
  parser.add_argument("--build-corpus", action="store_true", help="Write seo_optimizer_corpus.txt")
  parser.add_argument("--merge-corpus", action="store_true", help="Append SEO corpus to data/corpus.txt")
  parser.add_argument("--epochs", type=int, default=0, help="Run train.py after import with N epochs")
  args = parser.parse_args()

  entries = generate_entries()
  print(f"Generated {len(entries)} SEO optimizer training entries")

  if args.dry_run:
    return

  SEO_KB.parent.mkdir(parents=True, exist_ok=True)
  with SEO_KB.open("w", encoding="utf-8") as f:
    for row in entries:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  print(f"Wrote {SEO_KB}")

  existing = _load_jsonl(MAIN_KB)
  existing_q = {(r.get("q") or "").strip().lower() for r in existing}
  to_add = [r for r in entries if (r.get("q") or "").strip().lower() not in existing_q]
  if to_add:
    with MAIN_KB.open("a", encoding="utf-8") as f:
      for row in to_add:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended {len(to_add)} entries to {MAIN_KB}")
  else:
    print("No new entries for main knowledge base")

  if args.build_corpus or args.merge_corpus:
    corpus_text = build_corpus_pairs(entries)
    SEO_CORPUS.write_text(corpus_text, encoding="utf-8")
    print(f"Wrote {SEO_CORPUS} ({corpus_text.count('User:')} dialogue pairs)")
    if args.merge_corpus and corpus_text.strip():
      with MAIN_CORPUS.open("a", encoding="utf-8") as f:
        f.write("\n" + corpus_text)
      print(f"Merged SEO corpus into {MAIN_CORPUS}")

  if args.epochs > 0:
    corpus = MAIN_CORPUS if args.merge_corpus else SEO_CORPUS
    cmd = [
      sys.executable,
      str(PROJECT_ROOT / "scripts" / "train.py"),
      "--corpus", str(corpus),
      "--epochs", str(args.epochs),
      "--seq-len", "128",
    ]
    print("Training custom model:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

  print("Done. Restart the server to reload knowledge.")


if __name__ == "__main__":
  main()
