#!/usr/bin/env python3
"""Generate strong API key(s) for this service.

These are the keys your platform/frontend sends to THIS API as
`Authorization: Bearer <key>`. Generate once, then use as a fixed value:
  - put it in this API's API_KEY env var (comma-separate for multiple), and
  - paste the SAME key into your app's AI Credentials -> Custom -> API Key.

Usage:
  python scripts/generate_key.py        # one key
  python scripts/generate_key.py 3      # three keys (one per client/app)
"""

from __future__ import annotations

import secrets
import sys


def generate(n: int = 1) -> list[str]:
  return [secrets.token_urlsafe(36) for _ in range(max(1, n))]


def main() -> None:
  try:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
  except ValueError:
    count = 1
  keys = generate(count)
  for k in keys:
    print(k)
  if count > 1:
    print("\nAs one API_KEY value (comma-separated):")
    print(",".join(keys))


if __name__ == "__main__":
  main()
