"""SEO content enrichment — FAQs, local SEO, depth, schema, quality validation (all topics)."""

from __future__ import annotations

import re
from typing import Any, Callable
from urllib.parse import quote

# Major cities + common neighborhoods for local SEO extraction
_CITY_ALIASES: dict[str, list[str]] = {
  "bangalore": ["Bengaluru", "Koramangala", "Indiranagar", "Whitefield", "Electronic City", "MG Road", "HSR Layout"],
  "mumbai": ["Andheri", "Bandra", "Juhu", "Powai", "Colaba"],
  "delhi": ["Connaught Place", "Saket", "Dwarka", "Karol Bagh"],
  "hyderabad": ["Hitech City", "Banjara Hills", "Gachibowli"],
  "chennai": ["T Nagar", "Anna Nagar", "OMR"],
  "pune": ["Koregaon Park", "Hinjewadi", "Viman Nagar"],
  "kolkata": ["Park Street", "Salt Lake", "New Town"],
  "new york": ["Manhattan", "Brooklyn", "Queens"],
  "london": ["Westminster", "Camden", "Shoreditch"],
  "dubai": ["Marina", "Downtown", "Jumeirah"],
}

_GLOBAL_CITIES = frozenset(_CITY_ALIASES.keys()) | {k.title() for k in _CITY_ALIASES}

_SEMANTIC_BY_PROFILE: dict[str, list[str]] = {
  "adult_services": [
    "Privacy", "Discretion", "Companionship", "Agency", "Independent providers",
    "Booking", "Client expectations", "Availability", "Reviews", "Transparency",
  ],
  "local_services": ["Local providers", "Service area", "Booking", "Reviews", "Pricing", "Availability"],
  "automotive": ["Specifications", "Reliability", "Maintenance", "Fuel economy", "Safety features"],
  "security_escort": ["Risk assessment", "Protocol", "Convoy", "VIP protection", "Route planning"],
}

_DOMAIN_SEMANTIC: dict[str, list[str]] = {
  "tech": ["API", "Integration", "Performance", "Security", "Scalability", "Documentation", "Best practices"],
  "fitness": ["Recovery", "Consistency", "Form", "Progression", "Nutrition", "Hydration"],
  "business": ["ROI", "Strategy", "Conversion", "Audience", "Competition", "Pricing"],
  "health": ["Wellness", "Prevention", "Lifestyle", "Evidence-based", "Consultation"],
  "enterprise": ["Implementation", "Integration", "Compliance", "Workflow", "Training"],
  "general": ["Benefits", "Best practices", "Comparison", "Pricing", "Safety", "Reviews"],
}

_FAQ_QUESTION_PATTERNS: list[tuple[str, Callable[..., str]]] = []  # filled below


def _clip(text: str, n: int) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= n else t[: n - 3].rstrip() + "..."


def _count_words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _shingle_sim(a: str, b: str) -> float:
  wa = set(re.findall(r"\w+", a.lower()))
  wb = set(re.findall(r"\w+", b.lower()))
  if not wa or not wb:
    return 0.0
  return len(wa & wb) / len(wa | wb)


def extract_locations(topic: str, keywords: list[str]) -> list[str]:
  text = f"{topic} {' '.join(keywords)}"
  low = text.lower()
  found: list[str] = []
  seen: set[str] = set()
  for city_key, areas in _CITY_ALIASES.items():
    if city_key in low or city_key.title() in text or (city_key == "bangalore" and "bengaluru" in low):
      label = "Bangalore" if city_key == "bangalore" else city_key.title()
      if label.lower() not in seen:
        seen.add(label.lower())
        found.append(label)
      for area in areas:
        if area.lower() not in seen:
          seen.add(area.lower())
          found.append(area)
  for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", text):
    name = m.group(1)
    if name.lower() in _GLOBAL_CITIES and name.lower() not in seen:
      seen.add(name.lower())
      found.append(name)
  return found[:12]


def classify_intents_extended(
  topic: str,
  keywords: list[str],
  locations: list[str],
  base_intent: str,
) -> dict[str, Any]:
  intents: list[str] = []
  if locations:
    intents.append("local")
  if base_intent in ("commercial", "transactional"):
    intents.append("commercial")
  if base_intent in ("informational", "navigational") or not intents:
    intents.append("informational")
  if "commercial" not in intents and any(w in f"{topic} {' '.join(keywords)}".lower() for w in (
    "price", "cost", "best", "compare", "review", "agency", "service", "booking",
  )):
    intents.append("commercial")
  primary = intents[0] if intents else "informational"
  return {
    "primary": primary,
    "all": list(dict.fromkeys(intents)),
    "local": bool(locations),
    "commercial": "commercial" in intents,
    "informational": "informational" in intents,
  }


def extract_semantic_entities(
  topic: str,
  keywords: list[str],
  profile: str,
  domain: str,
  facts: list[Any],
) -> list[str]:
  entities: list[str] = []
  seen: set[str] = set()
  for pool in (
    _SEMANTIC_BY_PROFILE.get(profile, []),
    _DOMAIN_SEMANTIC.get(domain, _DOMAIN_SEMANTIC["general"]),
  ):
    for e in pool:
      k = e.lower()
      if k not in seen:
        seen.add(k)
        entities.append(e)
  for f in facts[:8]:
    text = getattr(f, "text", str(f))
    for m in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,2}\b", text):
      if len(m) > 3 and m.lower() not in seen:
        seen.add(m.lower())
        entities.append(m)
  for kw in keywords:
    if len(kw) > 3 and kw.lower() not in seen:
      seen.add(kw.lower())
      entities.append(kw.title() if kw.islower() else kw)
  return entities[:20]


def _faq_subject(ctx: dict[str, Any]) -> str:
  topic, primary = ctx["topic"], ctx["primary"]
  locations = {loc.lower() for loc in ctx.get("locations") or []}
  if primary.lower() in locations or (topic and primary.lower() in topic.lower() and len(topic) > len(primary)):
    return topic
  if len((topic or "").split()) > 5 and primary and len(primary.split()) <= 4:
    return primary.title() if primary.islower() else primary
  return primary or topic


def _answer_what_is(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  topic, profile = ctx["topic"], ctx["profile"]
  loc = ctx["locations"][0] if ctx.get("locations") else ""
  if profile in ("adult_services", "local_services"):
    base = (
      f"{label} generally involves pre-arranged companionship with agreed expectations "
      "and emphasis on privacy and professionalism."
    )
    if loc:
      base += f" In {loc}, providers may serve specific neighborhoods with varying availability."
    return base
  facts = ctx.get("facts") or []
  if facts:
    return _clip(getattr(facts[0], "text", str(facts[0])), 320)
  return (
    f"{label} refers to the practices, tools, and knowledge involved in {topic}. "
    "It includes core concepts, common use cases, and practical considerations for beginners and professionals."
  )


def _answer_how_works(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  topic, profile = ctx["topic"], ctx["profile"]
  if profile in ("adult_services", "local_services"):
    return (
      "Customers typically contact a provider or agency, confirm availability, discuss arrangements, "
      "and agree on pricing and meeting details before the appointment."
    )
  if profile == "tech" or "developer" in topic.lower() or "api" in topic.lower():
    return (
      f"Teams implement {label} through planning, integration, testing, and iteration. "
      "Workflows usually include setup, configuration, deployment, and ongoing monitoring."
    )
  return (
    f"The {label} process usually starts with research and planning, followed by setup, "
    f"execution, and refinement. Success depends on clear goals, the right tools, and consistent follow-through."
  )


def _answer_is_safe(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  profile = ctx["profile"]
  if profile in ("adult_services", "local_services"):
    return (
      "Safety depends on verifying providers, using reputable agencies, protecting personal information, "
      "and following local laws. Avoid sharing sensitive data before confirming credentials."
    )
  return (
    f"Safety with {label} depends on following best practices, using trusted sources, "
    "verifying credentials, and applying recommended guidelines from industry standards."
  )


def _answer_how_much(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  profile = ctx["profile"]
  if profile in ("adult_services", "local_services", "commercial"):
    return (
      "Pricing varies depending on duration, location, and service arrangements. "
      "Transparent quotes should be requested in advance before confirming any booking."
    )
  return (
    f"Costs for {label} vary by scope, provider, region, and complexity. "
    "Compare quotes, check what is included, and budget for setup, maintenance, and support."
  )


def _answer_before_choosing(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  locs = ", ".join(ctx.get("locations", [])[:3])
  loc_note = f" For local options in {locs}, verify service areas and reviews." if locs else ""
  return (
    f"Before choosing {label}, define your goals, compare reputable options, read reviews, "
    f"and confirm pricing, policies, and support.{loc_note}"
  )


def _answer_who_uses(ctx: dict[str, Any]) -> str:
  label = _faq_subject(ctx)
  topic = ctx["topic"]
  return (
    f"{label} is used by individuals and organizations that need reliable, structured "
    f"approaches to {topic}, from beginners learning fundamentals to experienced users optimizing outcomes."
  )


def _answer_local_areas(ctx: dict[str, Any]) -> str:
  topic = ctx["topic"]
  locs = ctx.get("locations") or []
  if not locs:
    return f"Service areas for {topic} depend on provider coverage and local demand."
  city = locs[0]
  areas = ", ".join(locs[1:6]) if len(locs) > 1 else ", ".join(locs[:5])
  return (
    f"In {city}, popular areas for {topic} include {areas}. "
    "Availability, pricing, and provider options can vary by neighborhood."
  )


_FAQ_HANDLERS: list[tuple[re.Pattern[str], Callable[[dict[str, Any]], str]]] = [
  (re.compile(r"^what is\b", re.I), _answer_what_is),
  (re.compile(r"how (?:does|do)\b", re.I), _answer_how_works),
  (re.compile(r"which areas|most relevant", re.I), _answer_local_areas),
  (re.compile(r"\bsafe\b", re.I), _answer_is_safe),
  (re.compile(r"how much|cost|pricing", re.I), _answer_how_much),
  (re.compile(r"before (?:choosing|using|booking)", re.I), _answer_before_choosing),
  (re.compile(r"who typically|who uses|who should", re.I), _answer_who_uses),
]


def _build_faq_questions(topic: str, primary: str, locations: list[str], seed: int) -> list[str]:
  label = topic if locations and primary.lower() in {loc.lower() for loc in locations} else (primary or topic)
  qs = [
    f"What is {label}?",
    f"How does {label} work?",
    f"Is {label} safe?",
    f"How much does {label} cost?",
    f"What should I know before choosing {label}?",
  ]
  if locations:
    qs.insert(2, f"Which areas in {locations[0]} are most relevant for {topic}?")
  if seed % 2 == 0:
    qs.append(f"Who typically uses {label}?")
  return qs[:6]


def generate_unique_faqs(
  topic: str,
  primary: str,
  keywords: list[str],
  facts: list[Any],
  *,
  profile: str,
  domain: str,
  intents: dict[str, Any],
  locations: list[str],
  seed: int,
) -> list[dict[str, str]]:
  ctx: dict[str, Any] = {
    "topic": topic,
    "primary": primary or topic,
    "keywords": keywords,
    "profile": profile,
    "domain": domain,
    "locations": locations,
    "facts": facts,
    "intents": intents,
    "seed": seed,
  }
  questions = _build_faq_questions(topic, primary or topic, locations, seed)
  faqs: list[dict[str, str]] = []
  used_answers: list[str] = []

  for q in questions:
    ans = ""
    for pat, handler in _FAQ_HANDLERS:
      if pat.search(q):
        ans = handler(ctx)
        break
    if not ans:
      ans = (
        f"Regarding {q.rstrip('?')}, research {primary} in the context of {topic} "
        "using verified sources and compare options that match your goals."
      )
    for prev in used_answers:
      if _shingle_sim(ans, prev) > 0.72:
        ans = ans + f" Focus specifically on {topic} and your local requirements."
        break
    used_answers.append(ans)
    faqs.append({"question": q, "answer": _clip(ans, 400)})

  dup_count = sum(
    1 for i, a in enumerate(used_answers)
    for j, b in enumerate(used_answers) if i < j and _shingle_sim(a, b) > 0.8
  )
  if dup_count > 0:
    faqs = _regenerate_duplicate_faqs(faqs, ctx)
  return faqs


def _regenerate_duplicate_faqs(
  faqs: list[dict[str, str]],
  ctx: dict[str, Any],
) -> list[dict[str, str]]:
  variants = [
    " Consider verified reviews and transparent policies.",
    " Compare at least two reputable options before deciding.",
    " Request written confirmation of pricing and terms.",
    " Check availability for your preferred date and location.",
    " Prioritize providers with clear communication channels.",
  ]
  seen: list[str] = []
  out: list[dict[str, str]] = []
  vi = 0
  for item in faqs:
    ans = item["answer"]
    while any(_shingle_sim(ans, s) > 0.8 for s in seen):
      ans = ans.rstrip(".") + variants[vi % len(variants)]
      vi += 1
    seen.append(ans)
    out.append({"question": item["question"], "answer": _clip(ans, 400)})
  return out


def optimize_title(topic: str, primary: str, intents: dict[str, Any], profile: str) -> str:
  subtitle_parts: list[str] = []
  if intents.get("local"):
    subtitle_parts.append("Local Guide")
  if profile in ("adult_services", "local_services"):
    subtitle_parts.extend(["Safety", "Pricing", "Booking Tips"])
  elif intents.get("commercial"):
    subtitle_parts.extend(["Comparison", "Pricing", "Tips"])
  else:
    subtitle_parts.extend(["Guide", "Tips", "Best Practices"])
  subtitle = ", ".join(subtitle_parts[:4])
  base = topic.strip() if topic else primary
  title = f"{base}: {subtitle}"
  return title[:70] if len(title) <= 70 else title[:67].rsplit(" ", 1)[0] + "..."


def optimize_meta_description(
  title: str,
  article: str,
  topic: str,
  primary: str,
  locations: list[str],
) -> str:
  intro = ""
  for line in article.splitlines():
    line = line.strip()
    if line and not line.startswith("#") and not line.startswith(">") and not line.startswith("|"):
      intro = re.sub(r"\*+", "", line)
      break
  if not intro:
    intro = f"Complete guide to {topic} covering {primary}."
  loc_bit = f" Areas: {', '.join(locations[:3])}." if locations else ""
  meta = f"{intro}{loc_bit} Expert tips, pricing factors, and FAQs."
  meta = re.sub(r"\s+", " ", meta).strip()
  if len(meta) > 160:
    meta = meta[:157].rsplit(" ", 1)[0] + "..."
  elif len(meta) < 120:
    meta = _clip(f"{meta} Learn options, safety, and booking best practices.", 160)
  return meta


def build_full_schema(
  title: str,
  meta: str,
  article: str,
  faqs: list[dict[str, str]],
  topic: str,
  slug: str,
) -> dict[str, Any]:
  headings = re.findall(r"^##\s+(.+)$", article, re.M)
  breadcrumb = {
    "@type": "BreadcrumbList",
    "itemListElement": [
      {"@type": "ListItem", "position": 1, "name": "Home", "item": "/"},
      {"@type": "ListItem", "position": 2, "name": topic, "item": f"/{slug}"},
    ],
  }
  if headings:
    breadcrumb["itemListElement"].append({
      "@type": "ListItem", "position": 3, "name": headings[0], "item": f"/{slug}#{quote(headings[0].lower())}",
    })
  article_schema = {
    "@type": "Article",
    "headline": title,
    "description": meta,
    "articleSection": headings[:8],
  }
  faq_schema = {
    "@type": "FAQPage",
    "mainEntity": [
      {
        "@type": "Question",
        "name": f["question"],
        "acceptedAnswer": {"@type": "Answer", "text": f["answer"]},
      }
      for f in faqs[:8]
    ],
  }
  return {
    "recommended_types": ["Article", "FAQPage", "BreadcrumbList"],
    "jsonld": {
      "@context": "https://schema.org",
      "@graph": [article_schema, faq_schema, breadcrumb],
    },
    "jsonld_hint": article_schema,
  }


def build_factor_table(topic: str, profile: str, seed: int) -> str:
  if profile in ("adult_services", "local_services"):
    rows = [
      ("Duration", "High"),
      ("Location", "High"),
      ("Availability", "High"),
      ("Experience", "Medium"),
      ("Reviews", "Medium"),
    ]
  elif profile == "automotive":
    rows = [
      ("Reliability", "High"),
      ("Fuel economy", "Medium"),
      ("Maintenance cost", "High"),
      ("Safety rating", "High"),
    ]
  else:
    rows = [
      ("Complexity", "High"),
      ("Cost", "Medium"),
      ("Time to implement", "High"),
      ("Expertise required", "Medium"),
      ("ROI potential", "High"),
    ]
  lines = ["| Factor | Influence |", "| --- | --- |"]
  for factor, influence in rows:
    lines.append(f"| {factor} | {influence} |")
  return "\n".join(lines)


def suggest_internal_links(topic: str, profile: str, outline: list[dict[str, str]]) -> list[dict[str, str]]:
  if profile in ("adult_services", "local_services"):
    suggestions = [
      ("Safety Guide", f"/guides/{_slug(topic)}-safety"),
      ("Privacy Tips", f"/guides/privacy-tips"),
      ("Booking Etiquette", f"/guides/booking-etiquette"),
      ("Companion Services Overview", f"/guides/companion-services"),
    ]
  else:
    headings = [o["text"] for o in outline if o.get("level") == "h2"][:4]
    suggestions = [(h, f"/guides/{_slug(h)}") for h in headings if h.lower() not in ("introduction", "conclusion")]
    if len(suggestions) < 3:
      suggestions.extend([
        (f"{topic} Overview", f"/guides/{_slug(topic)}"),
        ("Best Practices", f"/guides/best-practices"),
        ("FAQ", f"/guides/{_slug(topic)}-faq"),
      ])
  return [{"anchor_text": a, "url": u, "reason": "related topic cluster"} for a, u in suggestions[:6]]


def _slug(text: str) -> str:
  s = re.sub(r"[^\w\s-]", "", (text or "").lower())
  return re.sub(r"[-\s]+", "-", s).strip("-")[:60] or "guide"


_H3_EXPANSIONS: dict[str, list[str]] = {
  "Pricing Factors": ["Duration", "Location", "Availability", "Special requests"],
  "Pricing and Value Factors": ["Cost drivers", "Comparison tips", "Hidden fees", "Value metrics"],
  "Step-by-Step Guide": ["Preparation", "Execution", "Review", "Optimization"],
  "How It Works": ["Core workflow", "Tools involved", "Common pitfalls", "Success metrics"],
  "Safety and Privacy Considerations": ["Verification", "Data protection", "Meeting protocols", "Red flags"],
  "Booking Process": ["Initial inquiry", "Confirmation", "Payment terms", "Cancellation policy"],
}


def expand_article_depth(
  article: str,
  outline: list[dict[str, str]],
  topic: str,
  primary: str,
  *,
  target_words: int,
  semantic_entities: list[str],
  locations: list[str],
  profile: str,
) -> str:
  if _count_words(article) >= max(600, int(target_words * 0.75)):
    return _inject_semantic_entities(article, semantic_entities, locations)

  sections: list[str] = []
  blocks = re.split(r"\n(?=## )", article)
  for block in blocks:
    sections.append(block.strip())
    m = re.match(r"##\s+(.+)$", block.strip().split("\n")[0] if block.strip() else "")
    if not m:
      continue
    heading = m.group(1).strip()
    h3s = _H3_EXPANSIONS.get(heading, [])
    if not h3s and "pricing" in heading.lower():
      h3s = _H3_EXPANSIONS["Pricing Factors"]
    extra = ""
    for h3 in h3s[:4]:
      extra += f"\n\n### {h3}\n\n"
      if profile in ("adult_services", "local_services"):
        extra += (
          f"When evaluating {h3.lower()} for {topic}, confirm policies with the provider, "
          "compare options, and document agreed terms for transparency."
        )
      else:
        extra += (
          f"For {primary}, {h3.lower()} plays a key role in successful {topic} outcomes. "
          "Apply industry best practices and measure results over time."
        )
      if locations and h3.lower() == "location":
        extra += f" Popular areas include {', '.join(locations[:5])}."
    if extra:
      sections.append(extra.strip())

  expanded = "\n\n".join(s for s in sections if s)
  headings = re.findall(r"^##\s+(.+)$", expanded, re.M)
  pad_idx = 0
  while _count_words(expanded) < max(750, int(target_words * 0.8)) and pad_idx < 16:
    h = headings[pad_idx % len(headings)] if headings else topic
    expanded += (
      f"\n\n### Deeper dive: {h}\n\n"
      f"When working with **{primary}** for **{topic}**, focus on proven methods, measurable outcomes, "
      "and continuous improvement. Document decisions, compare alternatives, and validate assumptions "
      "with reliable references before scaling your approach."
    )
    if semantic_entities:
      ent = semantic_entities[pad_idx % len(semantic_entities)]
      expanded += f" Key concept: **{ent}**."
    pad_idx += 1

  if _count_words(expanded) < max(600, target_words // 2):
    expanded += (
      f"\n\n## Additional Considerations\n\n"
      f"Readers researching **{topic}** should also evaluate long-term value, support quality, "
      f"and alignment with goals related to {primary}. "
      + (f"Local context in {', '.join(locations[:3])} may affect availability and pricing. " if locations else "")
      + "Use the FAQ section for quick answers to common questions."
    )
  return _inject_semantic_entities(expanded, semantic_entities, locations)


def _inject_semantic_entities(
  article: str,
  semantic_entities: list[str],
  locations: list[str],
) -> str:
  if not semantic_entities and not locations:
    return article
  terms = semantic_entities[:8]
  if locations:
    terms = locations[:5] + terms
  if "## Key Terms" in article:
    return article
  block = "## Key Terms\n\n" + ", ".join(terms[:12]) + "."
  if "## Conclusion" in article:
    return article.replace("## Conclusion", block + "\n\n## Conclusion", 1)
  return article + "\n\n" + block


def apply_local_seo(
  article: str,
  topic: str,
  locations: list[str],
  keywords: list[str],
) -> tuple[str, list[str]]:
  if not locations:
    return article, keywords
  loc_para = (
    f"### Service Areas\n\n"
    f"For **{topic}**, relevant locations include {', '.join(locations[:8])}. "
    "Local availability, pricing, and provider options may vary by neighborhood."
  )
  if "### Service Areas" not in article and "## Introduction" in article:
    article = article.replace("## Introduction", "## Introduction\n\n" + loc_para + "\n\n", 1)
  kw_extended = list(dict.fromkeys(keywords + [loc.lower() for loc in locations[:6]]))
  return article, kw_extended


def inject_internal_links_section(
  article: str,
  links: list[dict[str, str]],
) -> str:
  if not links or "## Related Guides" in article:
    return article
  lines = ["## Related Guides", ""]
  for link in links[:5]:
    lines.append(f"- [{link['anchor_text']}]({link['url']})")
  block = "\n".join(lines)
  if "## Conclusion" in article:
    return article.replace("## Conclusion", block + "\n\n## Conclusion", 1)
  return article + "\n\n" + block


def validate_and_fix_content(
  draft: dict[str, Any],
  *,
  target_words: int,
  confidence: float,
) -> dict[str, Any]:
  article = draft.get("content", {}).get("article", "")
  faqs = list(draft.get("faqs") or [])
  issues: list[str] = []
  actions: list[str] = []

  wc = _count_words(article)
  if wc < 600:
    issues.append("article_short")
    actions.append("expand_depth")

  dup_answers = 0
  for i, a in enumerate(faqs):
    for j, b in enumerate(faqs):
      if i < j and _shingle_sim(a.get("answer", ""), b.get("answer", "")) > 0.8:
        dup_answers += 1
  if dup_answers > 0:
    issues.append(f"duplicate_faq_answers:{dup_answers}")
    ctx = {
      "topic": draft.get("topic", ""),
      "primary": (draft.get("keywords") or {}).get("primary", ""),
      "profile": draft.get("content_profile", "general"),
      "domain": draft.get("domain", "general"),
      "locations": draft.get("locations") or [],
      "facts": [],
      "intents": draft.get("intents") or {},
      "seed": draft.get("variation_seed") or 0,
    }
    faqs = _regenerate_duplicate_faqs(faqs, ctx)
    actions.append("regenerate_faqs")

  faq_sims = [
    _shingle_sim(faqs[i].get("answer", ""), faqs[j].get("answer", ""))
    for i in range(len(faqs)) for j in range(i + 1, len(faqs))
  ]
  if faq_sims and max(faq_sims) > 0.8:
    issues.append("faq_similarity_high")
    actions.append("regenerate_faqs")

  if confidence < 0.8 and confidence > 0:
    issues.append("low_rag_confidence")

  draft["faqs"] = faqs
  draft["quality_actions"] = actions
  return {
    "issues": issues,
    "actions": actions,
    "word_count": wc,
    "duplicate_answer_count": dup_answers,
    "faq_max_similarity": round(max(faq_sims) * 100, 1) if faq_sims else 0,
    "passed": not issues or issues == ["low_rag_confidence"],
  }
