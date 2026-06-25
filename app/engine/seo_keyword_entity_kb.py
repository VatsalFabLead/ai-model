"""Entity knowledge base — brands, products, and disambiguation for SEO keywords."""

from __future__ import annotations

import re
from typing import Any

# brand/term → domain, type, aliases
ENTITY_KB: dict[str, dict[str, Any]] = {
  "sugar": {
    "name": "Sugar",
    "type": "brand",
    "domain": "Cosmetics",
    "category": "Beauty & Fashion",
    "aliases": ("sugar cosmetics", "sugar makeup", "sugar beauty"),
    "context_hints": ("beauty", "cosmetic", "makeup", "skincare", "product", "company"),
    "reject_hints": ("sweetener", "diabetes", "cane", "glucose"),
  },
  "flutter": {
    "name": "Flutter",
    "type": "technology",
    "domain": "Mobile App Development",
    "category": "Technology",
    "aliases": ("flutter app", "flutter development"),
    "context_hints": ("app", "development", "mobile", "dart"),
  },
  "openai": {
    "name": "OpenAI",
    "type": "brand",
    "domain": "Artificial Intelligence",
    "category": "Technology",
    "aliases": ("chatgpt", "gpt"),
    "context_hints": ("ai", "llm", "chatbot"),
  },
  "zomato": {
    "name": "Zomato",
    "type": "brand",
    "domain": "Food Delivery",
    "category": "Food",
    "aliases": ("zomato food delivery",),
    "context_hints": ("food", "restaurant", "delivery"),
  },
  "nykaa": {
    "name": "Nykaa",
    "type": "brand",
    "domain": "Beauty",
    "category": "Beauty & Fashion",
    "aliases": ("nykaa beauty",),
    "context_hints": ("beauty", "cosmetics", "makeup"),
  },
  "hipaa": {
    "name": "HIPAA",
    "type": "regulation",
    "domain": "Healthcare",
    "category": "Healthcare & Medical",
    "aliases": ("hipaa compliance",),
    "context_hints": ("healthcare", "medical", "patient", "phi"),
  },
  "telemedicine": {
    "name": "Telemedicine",
    "type": "service",
    "domain": "Telemedicine",
    "category": "Healthcare & Medical",
    "aliases": ("virtual care", "remote healthcare"),
    "context_hints": ("healthcare", "doctor", "consultation"),
  },
}

_AMBIGUOUS_TERMS: dict[str, list[dict[str, Any]]] = {
  "sugar": [
    {"domain": "Cosmetics", "requires_any": ("beauty", "cosmetic", "makeup", "skincare", "product company")},
    {"domain": "Organic Food", "requires_any": ("organic", "food", "sweetener", "cane")},
  ],
  "apple": [
    {"domain": "Technology", "requires_any": ("iphone", "mac", "software", "app")},
    {"domain": "Organic Food", "requires_any": ("fruit", "juice", "organic food")},
  ],
  "amazon": [
    {"domain": "Ecommerce", "requires_any": ("marketplace", "shop", "prime", "aws")},
    {"domain": "Travel", "requires_any": ("rainforest", "river")},
  ],
}


def _clean(text: str) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def lookup_entity(term: str) -> dict[str, Any] | None:
  key = term.lower().strip()
  if key in ENTITY_KB:
    return {**ENTITY_KB[key], "term": key}
  for k, rec in ENTITY_KB.items():
    if key in rec.get("aliases", ()):
      return {**rec, "term": k}
  return None


def run_named_entity_recognition(
  seed: str,
  docs: list[Any],
  *,
  known_phrases: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
  """Extract entities from seed, KB, and open-data titles."""
  text = _clean(seed).lower()
  found: list[dict[str, Any]] = []
  seen: set[str] = set()

  def add(name: str, phrase: str, entity_type: str, domain: str = "", source: str = "seed") -> None:
    key = name.lower()
    if key in seen:
      return
    seen.add(key)
    found.append({
      "name": name,
      "phrase": phrase,
      "type": entity_type,
      "domain": domain,
      "source": source,
      "cluster": domain or name,
    })

  if known_phrases:
    consumed: list[tuple[int, int]] = []
    for phrase, cluster in sorted(known_phrases, key=lambda x: -len(x[0])):
      idx = text.find(phrase)
      if idx < 0:
        continue
      end = idx + len(phrase)
      if any(not (end <= s or idx >= e) for s, e in consumed):
        continue
      consumed.append((idx, end))
      add(cluster, phrase, "topic", domain=cluster, source="phrase_catalog")

  for term, rec in ENTITY_KB.items():
    if term in text or any(a in text for a in rec.get("aliases", ())):
      add(rec["name"], term, rec.get("type", "entity"), domain=rec.get("domain", ""), source="entity_kb")

  tokens = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", seed)
  for tok in tokens:
    if len(tok) > 3:
      kb = lookup_entity(tok.split()[0].lower())
      if kb:
        add(kb["name"], tok.lower(), kb.get("type", "brand"), domain=kb.get("domain", ""), source="entity_kb")

  for doc in docs[:8]:
    title = getattr(doc, "title", "") or ""
    for w in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", title):
      if len(w) > 4 and w.lower() not in seen:
        add(w, w.lower(), "open_data", source=getattr(doc, "source", "rag"))

  return found[:24]


def disambiguate_entities(
  entities: list[dict[str, Any]],
  seed: str,
  context: dict[str, Any],
) -> list[dict[str, Any]]:
  """Resolve ambiguous terms using seed context and KB rules."""
  haystack = seed.lower()
  out: list[dict[str, Any]] = []

  for ent in entities:
    phrase = ent.get("phrase", ent.get("name", "")).lower()
    resolved = dict(ent)
    amb = _AMBIGUOUS_TERMS.get(phrase.split()[0])
    if amb:
      for rule in amb:
        if any(h in haystack for h in rule.get("requires_any", ())):
          resolved["domain"] = rule["domain"]
          resolved["disambiguated"] = True
          break
      kb = lookup_entity(phrase.split()[0])
      if kb and not resolved.get("disambiguated"):
        reject = kb.get("reject_hints", ())
        if any(r in haystack for r in reject):
          resolved["rejected"] = True
        elif any(h in haystack for h in kb.get("context_hints", ())):
          resolved["domain"] = kb.get("domain", resolved.get("domain"))
          resolved["disambiguated"] = True
    elif lookup_entity(phrase.split()[0]):
      kb = lookup_entity(phrase.split()[0])
      if kb:
        resolved["domain"] = kb.get("domain", resolved.get("domain"))
        resolved["disambiguated"] = True
    out.append(resolved)

  return [e for e in out if not e.get("rejected")]


def detect_brand(seed: str, entities: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
  """Brand detection after NER + disambiguation."""
  tokens = [t for t in re.findall(r"\w+", seed) if len(t) > 1]
  low = seed.lower()
  brand_name = ""
  is_brand = False
  domain = context.get("primary_domain", "")

  for ent in entities:
    if ent.get("type") in ("brand", "entity") and ent.get("domain"):
      brand_name = ent["name"]
      is_brand = True
      break

  generic = {
    "beauty", "product", "products", "company", "cosmetics", "makeup", "skincare",
    "software", "development", "services", "solutions",
  }
  if not brand_name:
    caps = [t for t in tokens if t[0].isupper() and t.lower() not in generic]
    if any(w in low for w in ("beauty", "cosmetic", "makeup", "skincare")) and tokens:
      brand_name = tokens[0].title() if tokens[0][0].isupper() else caps[0] if caps else ""
      is_brand = bool(brand_name)
    elif caps:
      brand_name = " ".join(caps[:2])
      is_brand = len(tokens) >= 2

  kb_hit = lookup_entity(tokens[0].lower()) if tokens else None
  if kb_hit and kb_hit.get("type") == "brand":
    brand_name = kb_hit["name"]
    is_brand = True
    domain = kb_hit.get("domain", domain)

  return {
    "brand_name": brand_name,
    "is_brand_seed": is_brand,
    "domain": domain,
    "entities": [e.get("name") for e in entities if e.get("name")],
    "kb_match": kb_hit["name"] if kb_hit else None,
  }


def kb_primary_domain(seed: str, entities: list[dict[str, Any]]) -> str | None:
  for ent in entities:
    if ent.get("domain") and ent.get("disambiguated"):
      return ent["domain"]
  for term in re.findall(r"\w+", seed.lower()):
    rec = lookup_entity(term)
    if rec and rec.get("domain"):
      hints = rec.get("context_hints", ())
      if not hints or any(h in seed.lower() for h in hints):
        return rec["domain"]
  return None
