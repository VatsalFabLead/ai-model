"""Live factual data for tool context (free APIs, no GPT/Claude/Gemini)."""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

_USER_AGENT = "NexusCustomModel/1.0 (https://github.com/VatsalFabLead/ai-model)"
TROY_OZ_GRAMS = 31.1034768
_KARAT = {24: 1.0, 22: 22 / 24, 18: 18 / 24}

_GOLD_RE = re.compile(
  r"\b(gold price|price of gold|gold rate|today gold|gold today|"
  r"gold price today|today'?s gold)\b",
  re.IGNORECASE,
)


def is_gold_price_query(text: str) -> bool:
  low = text.lower()
  if "gold" not in low:
    return False
  return bool(_GOLD_RE.search(text) or ("gold" in low and ("today" in low or "price" in low or "rate" in low)))


def _usd_locale(text: str) -> bool:
  low = text.lower()
  return any(w in low for w in ("usd", "dollar", "us price", "united states", "america"))


def _inr_locale(text: str) -> bool:
  low = text.lower()
  if _usd_locale(text):
    return False
  if any(w in low for w in ("india", "indian", "inr", "rupee", "₹", "surat", "mumbai", "delhi", "gujarat")):
    return True
  # Default to India retail format when locale is not specified (common user intent).
  return True


async def _spot_gold_usd_per_oz(client: httpx.AsyncClient) -> float | None:
  """Fetch international spot gold (USD per troy oz) from free public APIs."""
  endpoints = (
    "https://api.gold-api.com/price/XAU",
    "https://api.metals.live/v1/spot/gold",
  )
  for url in endpoints:
    try:
      r = await client.get(url, headers={"User-Agent": _USER_AGENT})
      r.raise_for_status()
      data = r.json()
      if isinstance(data, dict) and "price" in data:
        return float(data["price"])
      if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, (int, float)):
          return float(item)
        if isinstance(item, dict):
          for k in ("price", "spot", "gold", "value"):
            if k in item:
              return float(item[k])
      if isinstance(data, dict):
        for k in ("price", "spot", "gold"):
          if k in data:
            return float(data[k])
    except (httpx.HTTPError, ValueError, TypeError):
      continue
  return None


async def _usd_to_inr(client: httpx.AsyncClient) -> float | None:
  try:
    r = await client.get(
      "https://open.er-api.com/v6/latest/USD",
      headers={"User-Agent": _USER_AGENT},
    )
    r.raise_for_status()
    data = r.json()
    rate = data.get("rates", {}).get("INR")
    return float(rate) if rate else None
  except (httpx.HTTPError, ValueError, TypeError):
    return None


def _format_inr_gold(usd_per_oz: float, inr_per_usd: float) -> str:
  usd_per_gram_24k = usd_per_oz / TROY_OZ_GRAMS
  today = datetime.now(timezone.utc).strftime("%d %B %Y")
  lines = [f"## Gold Price Today ({today}, India — approximate)", ""]
  for k in (24, 22, 18):
    purity = _KARAT[k]
    inr_per_g = usd_per_gram_24k * inr_per_usd * purity
    inr_per_10g = inr_per_g * 10
    lines.append(f"- **{k}K Gold:** ~₹{inr_per_g:,.0f} per gram (₹{inr_per_10g:,.0f} per 10 g)")
  lines.extend([
    "",
    "These are **approximate** retail reference rates from international spot gold "
    "and USD/INR — actual jeweller prices vary by city, GST, and making charges.",
    "",
    "*Not financial advice.*",
  ])
  return "\n".join(lines)


def _format_usd_gold(usd_per_oz: float) -> str:
  usd_per_gram = usd_per_oz / TROY_OZ_GRAMS
  today = datetime.now(timezone.utc).strftime("%d %B %Y")
  return (
    f"## Gold Price Today ({today})\n\n"
    f"- **Spot gold:** ~${usd_per_oz:,.2f} per troy ounce\n"
    f"- **Per gram (24K reference):** ~${usd_per_gram:,.2f}\n\n"
    "Rates are approximate from public spot data. Jeweller prices may differ.\n\n"
    "*Not financial advice.*"
  )


async def fetch_gold_price_context(query: str) -> str | None:
  if not is_gold_price_query(query):
    return None
  async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
    spot = await _spot_gold_usd_per_oz(client)
    if not spot:
      return None
    if _inr_locale(query):
      inr = await _usd_to_inr(client)
      if inr:
        return _format_inr_gold(spot, inr)
    return _format_usd_gold(spot)
