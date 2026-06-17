#!/usr/bin/env python3
"""Download a FREE open-source GGUF model for local generation (llama.cpp).

The default is Qwen2.5-0.5B-Instruct (Apache-2.0) — a free, open-source model.
It is NOT GPT/Claude/Gemini and involves no external AI company or API.

Usage:
  python scripts/download_model.py                 # default 0.5B model
  python scripts/download_model.py --url <gguf_url> --out models/llm/model.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Free, open-source (Apache-2.0). ~0.4 GB. Good for low-RAM machines.
DEFAULT_URL = (
  "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/"
  "resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
)
DEFAULT_OUT = PROJECT_ROOT / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# A larger, higher-quality free option (needs more RAM, ~1 GB):
#   https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf


def download(url: str, out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  if out.exists():
    print(f"Model already exists: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return

  print(f"Downloading:\n  {url}\n-> {out}")
  tmp = out.with_suffix(out.suffix + ".part")
  with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    done = 0
    last_pct = -1
    with tmp.open("wb") as f:
      for chunk in r.iter_bytes(chunk_size=1 << 20):
        f.write(chunk)
        done += len(chunk)
        if total:
          pct = int(done * 100 / total)
          if pct != last_pct and pct % 5 == 0:
            print(f"  {pct}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
            last_pct = pct
  tmp.replace(out)
  print(f"Done: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
  parser = argparse.ArgumentParser(description="Download a free open-source GGUF model")
  parser.add_argument("--url", default=DEFAULT_URL)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  args = parser.parse_args()

  try:
    download(args.url, args.out)
  except httpx.HTTPError as exc:
    print(f"Download failed: {exc}")
    print("Tip: pick another GGUF file URL from Hugging Face and pass it with --url.")
    sys.exit(1)


if __name__ == "__main__":
  main()
