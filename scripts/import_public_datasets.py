#!/usr/bin/env python3
"""Import FREE public datasets into Nexus KB + training corpus.

Supported sources (license-checked):
  wikidata, gutenberg, arxiv, gooaq, datacommons  — fully free (default)
  stackexchange, fineweb, c4, openwebtext, commoncrawl — free with attribution (default)
  starcoder                            — gated, permissive GitHub code (--include-gated)
  pile, redpajama, laion, kaggle        — mixed licenses (--include-mixed)

No GPT, Claude, or Gemini. Samples only (safe for shared hosting).

Usage:
  python scripts/import_public_datasets.py --list
  python scripts/import_public_datasets.py --datasets wikidata,arxiv,gutenberg
  python scripts/import_public_datasets.py --all --samples 15 --build-corpus
  python scripts/import_public_datasets.py --all --include-gated --include-mixed --samples 5

Optional HF streaming (fineweb, starcoder, pile, redpajama):
  pip install datasets
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.public_datasets import (  # noqa: E402
  DATASETS,
  list_datasets,
  sample_dataset,
)

KNOWLEDGE_FILE = PROJECT_ROOT / "data" / "knowledge.jsonl"
CORPUS_FILE = PROJECT_ROOT / "data" / "corpus.txt"
PUBLIC_CORPUS = PROJECT_ROOT / "data" / "public_datasets_corpus.txt"
ATTRIBUTIONS = PROJECT_ROOT / "data" / "ATTRIBUTIONS.md"


def corpus_block(entries: list[dict[str, str]]) -> str:
  lines: list[str] = []
  for row in entries:
    lines.append(f"User: {row['q']}\nAssistant: {row['a']} <|eos|>\n")
  return "\n".join(lines)


def append_attributions(used: list[str]) -> None:
  lines = [
    "# Dataset attributions (Nexus custom model)",
    "",
    f"Last import: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    "",
    "Only free or free-with-attribution sources are imported by default.",
    "",
  ]
  for ds_id in used:
    spec = DATASETS[ds_id]
    lines.extend([
      f"## {spec.name}",
      f"- Tier: `{spec.tier}`",
      f"- License: {spec.license}",
      f"- Attribution: {spec.attribution}",
      f"- Notes: {spec.notes}",
      "",
    ])
  ATTRIBUTIONS.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser(description="Import free public datasets")
  parser.add_argument("--list", action="store_true", help="Show datasets and license tiers")
  parser.add_argument("--datasets", type=str, default="", help="Comma-separated dataset ids")
  parser.add_argument("--all", action="store_true", help="Import all allowed datasets")
  parser.add_argument("--samples", type=int, default=15, help="Rows per dataset (default 15)")
  parser.add_argument("--max-chars", type=int, default=1200, help="Max chars per answer")
  parser.add_argument("--only-free", action="store_true", default=True)
  parser.add_argument("--include-gated", action="store_true")
  parser.add_argument("--include-mixed", action="store_true")
  parser.add_argument("--build-corpus", action="store_true")
  parser.add_argument("--epochs", type=int, default=0)
  args = parser.parse_args()

  if args.list:
    print("Dataset registry (100% custom training — no third-party AI models):\n")
    for spec in DATASETS.values():
      print(f"  {spec.id:14} tier={spec.tier:18} license={spec.license}")
    print("\nDefault import: fully_free + free_attribution")
    print("Add --include-gated for StarCoderData, --include-mixed for Pile/RedPajama")
    return

  allowed = {d.id for d in list_datasets(
    only_free=args.only_free,
    include_gated=args.include_gated,
    include_mixed=args.include_mixed,
  )}

  if args.all:
    selected = sorted(allowed)
  elif args.datasets:
    selected = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
  else:
    # Sensible default: API-only free sources (no HF install required)
    selected = [d for d in ("wikidata", "gutenberg", "arxiv", "stackexchange") if d in allowed]

  blocked = [d for d in selected if d not in allowed]
  if blocked:
    print(f"Skipped (license tier not allowed): {', '.join(blocked)}")
    print("Use --include-gated or --include-mixed if you accept those terms.")
  selected = [d for d in selected if d in allowed and d in DATASETS]
  if not selected:
    print("No datasets selected.")
    sys.exit(1)

  existing: set[str] = set()
  if KNOWLEDGE_FILE.exists():
    for line in KNOWLEDGE_FILE.read_text(encoding="utf-8").splitlines():
      if line.strip():
        try:
          existing.add(json.loads(line).get("q", "").strip().lower())
        except json.JSONDecodeError:
          pass

  all_entries: list[dict[str, str]] = []
  used_sources: list[str] = []

  for ds_id in selected:
    print(f"Sampling {ds_id} ({args.samples} rows)...")
    try:
      rows = sample_dataset(ds_id, limit=args.samples, max_chars=args.max_chars)
    except Exception as exc:
      print(f"  WARN {ds_id}: {exc}")
      continue
    used_sources.append(ds_id)
    new = 0
    for row in rows:
      key = row["q"].strip().lower()
      if key and key not in existing:
        existing.add(key)
        all_entries.append({"q": row["q"], "a": row["a"]})
        new += 1
    print(f"  {ds_id}: fetched {len(rows)}, added {new} KB entries")

  if all_entries:
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with KNOWLEDGE_FILE.open("a", encoding="utf-8") as f:
      for row in all_entries:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Appended {len(all_entries)} entries -> {KNOWLEDGE_FILE}")
  else:
    print("No new KB entries (duplicates or fetch errors).")

  if used_sources:
    append_attributions(used_sources)
    print(f"Wrote attributions -> {ATTRIBUTIONS}")

  if args.build_corpus and all_entries:
    text = corpus_block(all_entries)
    PUBLIC_CORPUS.write_text(text, encoding="utf-8")
    with CORPUS_FILE.open("a", encoding="utf-8") as f:
      f.write("\n" + text)
    print(f"Merged corpus -> {CORPUS_FILE}")

  if args.epochs > 0:
    cmd = [
      sys.executable,
      str(PROJECT_ROOT / "scripts" / "train.py"),
      "--corpus",
      str(CORPUS_FILE),
      "--epochs",
      str(args.epochs),
      "--seq-len",
      "128",
    ]
    print("Training:", " ".join(cmd))
    subprocess.run(cmd, check=True)

  print("Done. Restart server to reload knowledge.")


if __name__ == "__main__":
  main()
