"""SEO optimizer enrichment — convert gaps into real sections, FAQs, and metadata."""

from __future__ import annotations

import re
from typing import Any

INTERNAL_SUGGESTION_MARKERS = (
  "consider covering",
  "add a section",
  "add a paragraph",
  "not found in current content",
  "expand **",
  "mention **",
  "strengthen coverage",
  "add a faq section",
  "add a conclusion",
  "currently thin",
  "featured snippets",
  "call to action",
  "match search intent",
)


def is_internal_suggestion(text: str) -> bool:
  low = (text or "").lower().strip()
  if not low:
    return True
  return any(m in low for m in INTERNAL_SUGGESTION_MARKERS)


def _term_in_text(term: str, text: str) -> bool:
  return bool(re.search(rf"\b{re.escape(term.lower())}\b", text.lower()))


def _clip(text: str, n: int = 400) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= n else t[: n - 3].rstrip() + "..."


def parse_term_from_gap(gap: dict[str, str]) -> str | None:
  sug = gap.get("suggestion", "")
  m = re.search(r"\*\*([^*]+)\*\*", sug)
  if m:
    return m.group(1).strip()
  if gap.get("type") == "coverage_gap":
    for part in re.split(r"covering\s+", sug, flags=re.I):
      if part.strip():
        return re.sub(r"[^\w\s-]", "", part).strip()[:60]
  return None


def extract_gap_terms(
  gaps: list[dict[str, str]],
  coverage_map: dict[str, Any],
  keywords: list[str],
  *,
  article: str = "",
) -> list[str]:
  terms: list[str] = []
  seen: set[str] = set()
  body = article or ""
  # User-supplied keywords missing from content are highest priority
  for kw in keywords[:12]:
    if not kw or len(kw.strip()) < 2:
      continue
    k = kw.strip().lower()
    if k in seen:
      continue
    if body and _term_in_text(kw, body):
      continue
    seen.add(k)
    terms.append(kw.strip())
  for term in coverage_map.get("missing_terms", []) or []:
    k = term.lower()
    if k not in seen and len(term) > 2 and not is_internal_suggestion(term):
      seen.add(k)
      terms.append(term)
  for kw in keywords[1:8]:
    k = kw.lower()
    if k not in seen and len(kw) > 2:
      seen.add(k)
      terms.append(kw)
  allowed_types = {"coverage_gap", "topic_gap", "keyword_gap"}
  for g in gaps:
    if g.get("type") not in allowed_types:
      continue
    parsed = parse_term_from_gap(g)
    if parsed and not is_internal_suggestion(parsed):
      k = parsed.lower()
      if k not in seen:
        seen.add(k)
        terms.append(parsed)
  return terms[:12]


def detect_topic_profile(topic: str, keywords: list[str], content: str) -> str:
  blob = f"{topic} {' '.join(keywords)} {content[:2000]}".lower()
  if "flutter" in blob or "dart" in blob:
    return "flutter"
  if re.search(r"\berp\b", blob) or "enterprise resource" in blob:
    return "erp"
  if any(w in blob for w in ("api", "software", "framework", "programming")):
    return "tech"
  if any(w in blob for w in ("marketing", "seo", "conversion", "email")):
    return "marketing"
  return "general"


_KNOWLEDGE: dict[str, dict[str, str]] = {
  "flutter": {
    "flutter": (
      "Flutter is Google's open-source UI framework for building natively compiled applications "
      "for mobile, web, and desktop from a single codebase."
    ),
    "dart": (
      "Flutter uses Dart, a programming language developed by Google. Dart enables high performance "
      "and supports modern programming features including sound null safety."
    ),
    "hot reload": (
      "One of Flutter's most popular features is Hot Reload, which allows developers to see code "
      "changes instantly without restarting the application."
    ),
    "state management": (
      "Flutter supports several state management approaches, including Provider, Riverpod, Bloc, "
      "and GetX, each suited to different app complexity levels."
    ),
    "provider": (
      "Provider is a popular Flutter state management solution that makes dependency injection "
      "and state sharing straightforward for widget trees."
    ),
    "riverpod": (
      "Riverpod is a reactive state management library for Flutter that improves testability "
      "and compile-time safety compared to earlier patterns."
    ),
    "cross-platform": (
      "Flutter enables cross-platform development, allowing teams to ship Android, iOS, web, "
      "and desktop apps with shared UI code."
    ),
    "widget": (
      "Everything in Flutter is a widget. The framework uses a rich set of composable widgets "
      "to build flexible, expressive user interfaces."
    ),
    "performance": (
      "Flutter compiles to native ARM code and uses the Skia graphics engine for smooth, "
      "high-performance rendering across platforms."
    ),
  },
  "marketing": {
    "email marketing": (
      "Email marketing uses targeted messages to nurture leads, drive conversions, and retain customers "
      "through newsletters, campaigns, and automated sequences."
    ),
    "conversions": (
      "Conversion optimization focuses on turning readers into customers through clear CTAs, "
      "relevant offers, and frictionless signup flows."
    ),
  },
  "general": {},
}


def _match_knowledge(profile: str, term: str) -> str | None:
  pool = _KNOWLEDGE.get(profile, {})
  low = term.lower()
  for key, body in pool.items():
    if key in low or low in key:
      return body
  return _KNOWLEDGE.get("general", {}).get(low)


def generate_section_for_term(
  term: str,
  topic: str,
  keywords: list[str],
  *,
  profile: str,
  facts: list[dict[str, Any]],
  seed: int,
) -> str:
  heading = term.strip().title() if term.islower() else term.strip()
  known = _match_knowledge(profile, term)
  if known:
    body = known
  else:
    body = ""
    low = term.lower()
    for f in facts:
      text = f.get("text", "")
      if low in text.lower() and not is_internal_suggestion(text):
        body = _clip(text, 360)
        break
    if not body:
      primary = keywords[0] if keywords else topic
      body = (
        f"{heading} is a key consideration for {topic}. "
        f"When evaluating {primary}, understanding {heading.lower()} helps readers make informed "
        "decisions, compare options, and apply best practices effectively."
      )
  return f"## {heading}\n\n{body}"


def fill_content_gaps(
  article: str,
  gaps: list[dict[str, str]],
  coverage_map: dict[str, Any],
  *,
  topic: str,
  keywords: list[str],
  facts: list[dict[str, Any]] | None = None,
  seed: int = 0,
) -> tuple[str, list[str]]:
  """Inject real H2 sections for missing terms — never paste gap suggestions."""
  facts = facts or []
  profile = detect_topic_profile(topic, keywords, article)
  terms = extract_gap_terms(gaps, coverage_map, keywords, article=article)
  added: list[str] = []
  blocks: list[str] = []

  for i, term in enumerate(terms):
    if _term_in_text(term, article):
      continue
    blocks.append(generate_section_for_term(
      term, topic, keywords, profile=profile, facts=facts, seed=seed + i,
    ))
    added.append(term)

  if not blocks:
    return article, added

  injection = "\n\n".join(blocks)
  if re.search(r"^##\s+conclusion\b", article, re.I | re.M):
    article = re.sub(
      r"^(##\s+conclusion\b)",
      injection + "\n\n\\1",
      article,
      count=1,
      flags=re.I | re.M,
    )
  elif re.search(r"^##\s+frequently asked questions\b", article, re.I | re.M):
    article = re.sub(
      r"^(##\s+frequently asked questions\b)",
      injection + "\n\n\\1",
      article,
      count=1,
      flags=re.I | re.M,
    )
  else:
    article = article.rstrip() + "\n\n" + injection

  return article.strip(), added


def regenerate_keyword_optimized_content(
  content: str,
  keywords: list[str],
  *,
  tone: str = "professional",
  seed: int = 0,
) -> tuple[str, list[str]]:
  """Rewrite content around user-supplied keywords — works with any topic/input."""
  if not content.strip():
    return content, []
  if not keywords:
    return content.strip(), []

  suggestions: list[str] = []
  primary = keywords[0].strip()
  topic = primary
  text = content.strip()
  profile = detect_topic_profile(topic, keywords, text)

  if not re.search(r"^#\s+", text, re.MULTILINE):
    title = primary.title() if len(primary) < 80 else _clip(primary, 70)
    text = f"# {title}\n\n{text}"
    suggestions.append("Added H1 title from primary keyword.")

  paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() and not p.startswith("#")]
  first_para = paras[0] if paras else text[:500]
  if primary and not _term_in_text(primary, first_para[:500]):
    opener = {
      "professional": f"**{primary}** is central to this guide. ",
      "casual": f"Let's talk about **{primary}** — ",
      "friendly": f"Here's a clear look at **{primary}**. ",
      "formal": f"This document addresses **{primary}** in detail. ",
    }.get(tone, f"**{primary}** — ")
    if paras:
      text = text.replace(first_para, opener + first_para, 1)
    else:
      text = f"{text}\n\n{opener.strip()}"
    suggestions.append(f"Wove primary keyword '{primary}' into the introduction.")

  for i, kw in enumerate(keywords[1:10]):
    kw = kw.strip()
    if not kw or _term_in_text(kw, text):
      continue
    block = generate_section_for_term(kw, topic, keywords, profile=profile, seed=seed + i + 7)
    text = text.rstrip() + "\n\n" + block
    suggestions.append(f"Added optimized section for keyword: {kw}")

  takeaways = build_key_takeaways(topic, keywords, text, [k for k in keywords if not _term_in_text(k, text)])
  if takeaways and "## Key Takeaways" not in text:
    block = "## Key Takeaways\n\n" + "\n".join(f"- {t}" for t in takeaways[:5]) + "\n"
    if re.search(r"^##\s+conclusion\b", text, re.I | re.M):
      text = re.sub(r"^(##\s+conclusion\b)", block + "\n\\1", text, count=1, flags=re.I | re.M)
    else:
      text = text.rstrip() + "\n\n" + block
    suggestions.append("Added keyword-aware key takeaways.")

  if primary and not _term_in_text(primary, text[-600:]):
    text = text.rstrip() + (
      f"\n\n## Summary\n\n"
      f"Effective **{primary}** content balances clarity, depth, and search intent. "
      f"Use the sections above to improve rankings and reader value.\n"
    )
    suggestions.append("Added summary reinforcing primary keyword.")

  return text.strip(), suggestions


def build_key_takeaways(
  topic: str,
  keywords: list[str],
  article: str,
  gap_terms: list[str],
) -> list[str]:
  profile = detect_topic_profile(topic, keywords, article)
  takeaways: list[str] = []
  seen: set[str] = set()

  if profile == "flutter":
    candidates = [
      "Flutter uses Dart for fast, modern app development.",
      "Hot Reload accelerates UI iteration during development.",
      "Provider and Riverpod simplify state management.",
      "Flutter supports cross-platform Android, iOS, web, and desktop apps.",
      "Widgets are the building blocks of every Flutter interface.",
    ]
    for c in candidates:
      k = c.lower()
      if k not in seen:
        seen.add(k)
        takeaways.append(c)

  for term in gap_terms[:6]:
    known = _match_knowledge(profile, term)
    if known:
      line = _clip(known, 100)
      if line.lower() not in seen:
        seen.add(line.lower())
        takeaways.append(line if line.endswith(".") else line + ".")

  for m in re.findall(r"^[\*\-]\s+(.+)$", article, re.M):
    line = re.sub(r"\*+", "", m).strip()
    if line and not is_internal_suggestion(line) and line.lower() not in seen:
      seen.add(line.lower())
      takeaways.append(_clip(line, 100))

  if len(takeaways) < 3:
    primary = keywords[0] if keywords else topic
    takeaways.append(f"{primary.title()} improves outcomes when applied with clear goals and best practices.")

  return takeaways[:5]


def _extract_intro_sentence(article: str, topic: str) -> str:
  skip_heads = {"key takeaways", "quick takeaways", "frequently asked questions", "related guides"}
  in_skip = False
  for line in article.splitlines():
    line = line.strip()
    if line.startswith("##"):
      in_skip = line.lstrip("#").strip().lower() in skip_heads
      continue
    if in_skip:
      continue
    if not line or line.startswith("#") or line.startswith(">") or line.startswith("|"):
      continue
    if line.startswith("-") or line.startswith("*"):
      continue
    if not is_internal_suggestion(line):
      return _clip(re.sub(r"\*+", "", line), 320)
  pool = _match_knowledge(detect_topic_profile(topic, [], article), topic)
  return pool or f"{topic} is covered in this guide with practical, search-aligned information."


def _section_answer(article: str, term: str, profile: str = "general") -> str:
  known = _match_knowledge(profile, term)
  low = term.lower()
  for heading in re.findall(r"^##\s+(.+)$", article, re.M):
    if heading.lower() == low or heading.lower().startswith(low + " ") or low == heading.lower().split()[0]:
      body = ""
      in_sec = False
      for line in article.splitlines():
        if re.match(rf"^##\s+{re.escape(heading)}\s*$", line, re.I):
          in_sec = True
          continue
        if in_sec and line.startswith("##"):
          break
        if in_sec and line.strip() and not line.startswith("#"):
          body += line.strip() + " "
      if body.strip() and not is_internal_suggestion(body):
        return _clip(body.strip(), 400)
  return known or ""


def generate_optimizer_faqs(
  topic: str,
  keywords: list[str],
  article: str,
  *,
  entities: list[str] | None = None,
  seed: int = 0,
) -> list[dict[str, str]]:
  """Topic-aware FAQs — no prompt leakage from gap suggestions."""
  primary = keywords[0] if keywords else topic
  profile = detect_topic_profile(topic, keywords, article)
  intro = _extract_intro_sentence(article, topic)
  faqs: list[dict[str, str]] = []
  seen: set[str] = set()

  def add(q: str, a: str) -> None:
    q = re.sub(r"\s+", " ", q.strip())
    a = _clip(a, 400)
    if not q or not a or is_internal_suggestion(q) or is_internal_suggestion(a):
      return
    if q.lower() in seen:
      return
    seen.add(q.lower())
    faqs.append({"question": q, "answer": a})

  label = primary.title() if primary else topic
  if profile == "flutter":
    add(f"What is {label}?", _KNOWLEDGE["flutter"]["flutter"])
    add("What programming language does Flutter use?", _KNOWLEDGE["flutter"]["dart"])
    add("What is Hot Reload?", _KNOWLEDGE["flutter"]["hot reload"])
    add("What is state management in Flutter?", _KNOWLEDGE["flutter"]["state management"])
    add(
      "Can Flutter build Android and iOS apps?",
      "Yes. Flutter supports cross-platform development for Android, iOS, web, and desktop from a shared codebase.",
    )
  else:
    # Prefer intent-led FAQs from real H2s / entities — avoid generic
    # "What is X / How does X work / Benefits of X" templates.
    for h2 in re.findall(r"^##\s+(.+)$", article, re.M):
      low = h2.lower().strip()
      if low in ("introduction", "conclusion", "frequently asked questions", "key takeaways", "related guides"):
        continue
      body = _section_answer(article, h2, profile)
      if len(body) < 40:
        continue
      if low.startswith(("what ", "how ", "why ", "when ", "where ")):
        q = h2 if h2.endswith("?") else f"{h2}?"
      else:
        q = f"Why does {h2} matter?"
      add(q, body)
      if len(faqs) >= 5:
        break
    if intro and len(faqs) < 3:
      add(f"What should readers know about {label}?", intro)
    if len(faqs) < 4:
      add(
        f"Why is {label} important?",
        _section_answer(article, keywords[1] if len(keywords) > 1 else label, profile)
        or intro
        or f"This article explains practical context around {topic}.",
      )

  for ent in (entities or [])[:3]:
    if len(faqs) >= 10:
      break
    if _term_in_text(ent, article):
      ans = _section_answer(article, ent, profile)
      if ans:
        add(f"What is {ent}?", ans)

  return faqs[:8]


def optimize_metadata_clean(
  topic: str,
  keywords: list[str],
  article: str,
  *,
  seed: int = 0,
) -> dict[str, str]:
  """Title and meta from real content only — never internal optimizer notes."""
  primary = keywords[0] if keywords else topic
  profile = detect_topic_profile(topic, keywords, article)

  h1 = ""
  m = re.search(r"^#\s+(.+)$", article, re.M)
  if m:
    h1 = re.sub(r"\s+", " ", m.group(1).strip())

  # Prefer real H1 over generic "Guide / Tips / Best Practices" templates
  if h1 and len(h1) >= 8:
    title = h1[:70] if len(h1) <= 70 else h1[:67].rsplit(" ", 1)[0] + "..."
  elif profile == "flutter":
    title = f"{topic.strip()}: Features, Benefits, and Best Practices"[:70]
  else:
    title = (topic.strip() or primary)[:70]

  intro = _extract_intro_sentence(article, topic)
  feature_bits: list[str] = []
  if profile == "flutter":
    feature_bits = ["Hot Reload", "state management", "Android and iOS development"]
  elif keywords:
    feature_bits = [k for k in keywords[1:4] if k and not is_internal_suggestion(k)]

  meta = intro
  if profile == "flutter" and feature_bits:
    meta = f"Learn {primary}, {', '.join(feature_bits[:3])}, and best practices for {topic}."
  elif feature_bits and intro:
    meta = intro
  else:
    meta = intro or f"Read about {primary}."
  meta = re.sub(r"\s+", " ", meta).strip()
  if len(meta) > 160:
    meta = meta[:157].rsplit(" ", 1)[0] + "..."
  elif len(meta) < 110 and intro:
    meta = _clip(intro, 160)

  if is_internal_suggestion(meta) or is_internal_suggestion(title):
    title = (h1 or primary or topic)[:70]
    meta = _clip(intro or f"Overview of {primary} based on the article content.", 160)

  return {"title": title, "meta_description": meta}


def filter_content_pools(sentences: list[str], bullets: list[str]) -> tuple[list[str], list[str]]:
  clean_s = [s for s in sentences if s and not is_internal_suggestion(s)]
  clean_b = [b for b in bullets if b and not is_internal_suggestion(b)]
  return clean_s, clean_b
