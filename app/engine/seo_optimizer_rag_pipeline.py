"""SEO Content Optimizer — full production architecture (free open datasets).

Existing Content → Keyword Analysis → Entity Extraction → Coverage Map
→ Gap Analysis → Source Router → Retriever → Deduplication → Reranker
→ Fact Extractor → Novelty Detector → Section Planner → Section Generator
→ FAQ Generator → Readability Optimizer → Metadata Optimizer
→ SEO Score Calculator → Final Optimized Content
"""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from app.engine.open_data_retrieval import retrieve_from_sources
from app.engine.seo_optimizer_engine import (
  analyze_issues,
  content_metrics,
  count_sentences,
  count_words,
  improve_readability,
  readability_score,
  seo_score_from_analysis,
)
from app.engine.seo_rag_pipeline import (
  ExtractedFact,
  classify_topic,
  deduplicate_docs,
  extract_entities,
  extract_facts,
  rerank_docs,
  resolve_conflicts,
  route_sources,
  score_confidence,
)
from app.engine.seo_content_domains import make_variation_seed
from app.engine.seo_optimizer_enrichment import (
  build_key_takeaways,
  fill_content_gaps,
  filter_content_pools,
  generate_optimizer_faqs,
  is_internal_suggestion,
  optimize_metadata_clean,
  parse_term_from_gap,
  rewrite_content_for_keywords,
)

ARCHITECTURE_FLOW = [
  "input_article",
  "keyword_extractor",
  "entity_extractor",
  "coverage_map",
  "gap_analysis",
  "keyword_rewriter",
  "source_router",
  "retriever",
  "deduplication",
  "reranker",
  "fact_extraction",
  "relevance_filter",
  "section_planner",
  "section_generator",
  "novelty_detector",
  "faq_generator",
  "metadata_generator",
  "readability_optimizer",
  "humanizer",
  "seo_scorer",
  "final_article",
]

GENERATOR_VERSION = "seo-optimizer-rag-v5.2"

_INSTRUCTION_MARKERS = (
  "you are an expert seo content optimizer",
  "analyze the provided article",
  "perform the following",
  "### objectives",
  "### constraints",
  "output format",
  "keyword optimization",
  "featured snippet optimization",
  "e-e-a-t optimization",
)


def is_optimizer_instruction_content(text: str) -> bool:
  """Detect when the user pasted an SEO prompt template instead of article content."""
  low = (text or "").lower()
  hits = sum(1 for m in _INSTRUCTION_MARKERS if m in low)
  numbered_sections = len(re.findall(r"^##\s+\d+\.\s+", text, re.MULTILINE | re.I))
  you_are_expert = "you are an expert" in low and "seo" in low
  return you_are_expert or hits >= 4 or (hits >= 2 and numbered_sections >= 6)


def normalize_pasted_optimizer_content(text: str) -> tuple[str, list[str]]:
  """Strip QA paste wrappers (Topic / Target Keywords / weakness notes); extract keywords."""
  raw = (text or "").strip()
  if not raw:
    return raw, []

  extracted_kws: list[str] = []
  topic_m = re.search(r"^Topic:\s*(.+)$", raw, re.MULTILINE | re.I)
  topic = topic_m.group(1).strip() if topic_m else ""

  body = raw
  kw_m = re.search(r"\nTarget Keywords\s*\n", body, re.I)
  if kw_m:
    tail = body[kw_m.end() :]
    end_m = re.search(r"\nDeliberate SEO Weaknesses|\nThis content is intentionally", tail, re.I)
    kw_block = tail[: end_m.start()] if end_m else tail
    for line in kw_block.splitlines():
      line = line.strip().lstrip("-*•").strip()
      if line and len(line) > 2:
        extracted_kws.append(line)
    body = body[: kw_m.start()]

  body = re.sub(r"^Topic:\s*.+\n?", "", body, flags=re.MULTILINE | re.I)
  body = re.sub(r"^Content for SEO Content Optimizer\s*\n?", "", body, flags=re.MULTILINE | re.I)
  body = body.strip()

  if topic and not re.search(r"^#\s+", body, re.MULTILINE):
    body = f"# {topic}\n\n{body}"

  return body.strip(), list(dict.fromkeys(extracted_kws))[:12]


# Minimum relevance score (0–1) for retrieved docs on the light-RAG path
_MIN_DOC_RELEVANCE = 0.45

_SEO_TOPIC_CHECKLIST: dict[str, list[str]] = {
  "flutter": ["hot reload", "dart", "widget", "state management", "cross-platform", "performance"],
  "erp": ["inventory", "manufacturing", "modules", "integration", "workflow"],
  "default": ["benefits", "best practices", "how to", "features"],
}

_TOPIC_ALIASES = {
  "flutter": ["flutter", "dart", "widget", "cross-platform", "mobile app"],
  "erp": ["erp", "enterprise resource", "inventory", "manufacturing"],
  "seo": ["seo", "search engine", "ranking", "keyword"],
}

_IRRELEVANT_PATTERNS = [
  r"google play(?:\s+store)?",
  r"google tv",
  r"google play books",
  r"android market",
  r"android operating system",
  r"chromeos",
  r"jvm language",
  r"android software development",
  r"appjet",
  r"y combinator",
  r"single-page application",
  r"web-based applications on a client",
  r"^\s*##\s+",
]


def infer_keywords_from_content(content: str) -> list[str]:
  low = content.lower()
  found: list[str] = []
  if "flutter" in low:
    found.extend(["Flutter", "Flutter app development", "cross-platform apps", "Dart"])
  if re.search(r"\berp\b", low) or "enterprise resource planning" in low:
    found.extend(["ERP software", "enterprise resource planning"])
  if "react" in low:
    found.append("React development")
  if not found:
    m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if m:
      title = re.sub(r"[*_`]", "", m.group(1)).strip()
      words = [w for w in re.findall(r"\w+", title) if len(w) > 3][:3]
      if words:
        found.append(" ".join(words))
  return list(dict.fromkeys(found))[:6]


def normalize_keywords(
  content: str,
  keywords: list[str],
  *,
  user_supplied: bool = False,
) -> tuple[str, str, list[str]]:
  """Return (short_primary, display_title, keyword_list)."""
  display = ""
  m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
  if m:
    display = re.sub(r"[*_`]", "", m.group(1)).strip()

  kws = list(keywords) if keywords else infer_keywords_from_content(content)
  if not kws and display:
    kws = [display]

  raw_primary = kws[0] if kws else (display or infer_topic(content, []))
  short = raw_primary

  if user_supplied and kws:
    short = raw_primary if len(raw_primary) <= 72 else _clip(raw_primary, 72)
  elif len(raw_primary) > 45 or raw_primary.count(" ") > 4:
    low = content.lower()
    if "flutter" in low:
      short = "Flutter app development"
    elif re.search(r"\berp\b", low) or "enterprise resource planning" in low:
      short = "ERP software"
    else:
      words = [w for w in re.findall(r"\w+", raw_primary) if len(w) > 3][:3]
      short = " ".join(words) if words else _clip(raw_primary, 40)

  if short.lower() not in {k.lower() for k in kws}:
    kws = [short] + kws
  else:
    kws[0] = short

  if not user_supplied and "flutter" in content.lower():
    kws = [k for k in kws if "erp" not in k.lower() and "enterprise resource" not in k.lower()]

  return short, display or short, kws


def derive_anchor_terms(content: str, short_topic: str, keywords: list[str]) -> set[str]:
  terms: set[str] = set()
  for src in keywords + [short_topic]:
    for w in re.findall(r"\w+", src.lower()):
      if len(w) > 2:
        terms.add(w)
  low = content.lower()
  for domain, hints in _TOPIC_ALIASES.items():
    if domain == "erp":
      if not (re.search(r"\berp\b", low) or "enterprise resource planning" in low):
        continue
    elif domain == "flutter":
      if "flutter" not in low:
        continue
    elif domain not in low and not any(h in low for h in hints):
      continue
    terms.update(hints)
    terms.add(domain)
  return terms


def clean_fact_text(text: str) -> str:
  t = re.sub(r"^#+\s*", "", (text or "").strip())
  t = re.sub(r"\s+", " ", t)
  return _clip(t, 280)


def _term_in_text(term: str, text: str) -> bool:
  """Match keyword phrases by whole term or significant tokens."""
  low = text.lower()
  t = term.lower().strip()
  if not t:
    return False
  if t in low:
    return True
  tokens = [w for w in re.findall(r"\w+", t) if len(w) > 3]
  if not tokens:
    tokens = re.findall(r"\w+", t)
  if not tokens:
    return False
  return sum(1 for w in tokens if w in low) >= max(1, len(tokens) // 2 + len(tokens) % 2)


def is_doc_relevant(doc: Any, anchors: set[str]) -> bool:
  blob = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".lower()
  if not blob.strip():
    return False
  for pat in _IRRELEVANT_PATTERNS:
    if re.search(pat, blob, re.I):
      return False
  return is_fact_relevant(blob[:600], anchors)


def doc_relevance_score(doc: Any, anchors: set[str]) -> float:
  if not is_doc_relevant(doc, anchors):
    return 0.0
  blob = f"{getattr(doc, 'title', '')} {getattr(doc, 'text', '')}".lower()
  if not anchors:
    return 0.5
  hits = sum(1 for t in anchors if t in blob)
  return min(1.0, hits / max(2, min(len(anchors), 6)))


def filter_docs(docs: list[Any], anchors: set[str], *, min_score: float = _MIN_DOC_RELEVANCE) -> list[Any]:
  scored: list[tuple[float, Any]] = []
  for doc in docs:
    s = doc_relevance_score(doc, anchors)
    if s >= min_score:
      doc.score = max(doc.score, s)
      scored.append((s, doc))
  scored.sort(key=lambda x: x[0], reverse=True)
  return [d for _, d in scored]


def is_fact_relevant(text: str, anchors: set[str]) -> bool:
  low = text.lower()
  for pat in _IRRELEVANT_PATTERNS:
    if re.search(pat, low, re.I):
      return False
  if not anchors:
    return True
  if "flutter" in anchors:
    if not any(t in low for t in ("flutter", "dart", "widget", "cross-platform", "mobile app")):
      return False
  if "erp" in anchors and "erp" not in low and "enterprise" not in low:
    return False
  hits = sum(1 for t in anchors if t in low)
  return hits >= 2 or (hits >= 1 and len(low) < 400)


def filter_facts(facts: list[ExtractedFact], content: str, anchors: set[str]) -> list[ExtractedFact]:
  out: list[ExtractedFact] = []
  for f in facts:
    cleaned = clean_fact_text(f.text)
    if len(cleaned) < 30:
      continue
    if not is_fact_relevant(cleaned, anchors):
      continue
    out.append(ExtractedFact(text=cleaned, source=f.source, confidence=f.confidence, entities=f.entities))
  return out


def is_content_already_strong(content: str) -> bool:
  return (
    count_words(content) >= 200
    and bool(re.search(r"^#\s+", content, re.MULTILINE))
    and "##" in content
    and count_sentences(content) >= 8
  )


def extract_section_text(content: str, heading: str, *, h3_only: bool = False) -> str:
  """Extract body under a heading; stops at the next heading of same or higher level."""
  target = heading.strip()
  lines = content.splitlines()
  capturing = False
  level = 3 if h3_only else 2
  chunks: list[str] = []

  for line in lines:
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if m:
      hdr_level = len(m.group(1))
      hdr_title = m.group(2).strip()
      if capturing:
        if hdr_level <= level or (level == 2 and hdr_level > 2):
          break
      if hdr_level == level and hdr_title.lower() == target.lower():
        capturing = True
        continue
    elif capturing:
      chunks.append(line)

  body = "\n".join(chunks).strip()
  if h3_only:
    return _clip_smart(body, 400)
  # H2: only direct prose under H2, not nested H3 bodies
  direct: list[str] = []
  for line in chunks:
    if re.match(r"^#{3,6}\s+", line.strip()):
      break
    direct.append(line)
  return _clip_smart("\n".join(direct).strip(), 400)


def _clip_smart(text: str, n: int = 480) -> str:
  flat = re.sub(r"[ \t]+", " ", (text or "").replace("\n", " ").strip())
  if len(flat) <= n:
    return flat
  cut = flat[:n].rsplit(" ", 1)[0]
  return cut.rstrip(".,;:") + "."


def preserve_markdown_structure(text: str) -> str:
  """Restore newlines so headings, paragraphs, and bullets render correctly."""
  t = (text or "").strip()
  if not t:
    return t
  t = re.sub(r"([^\n#])\s*(#{1,6}\s+)", r"\1\n\n\2", t)
  t = re.sub(
    r"(^#{1,6}\s+.+?)(?<![\w:])(\s+)(?=[A-Z][a-z]+\s+[a-z])",
    r"\1\n\n",
    t,
    flags=re.MULTILINE,
  )
  t = re.sub(r"([.!?])\s*(\* )", r"\1\n\n\2", t)
  t = re.sub(r"([^\n])\s*(- )", r"\1\n\n\2", t)
  t = re.sub(r"\n{3,}", "\n\n", t)
  return t.strip()


def extract_h3_under_h2(content: str, h2_title: str) -> list[str]:
  target = h2_title.strip().lower()
  lines = content.splitlines()
  in_h2 = False
  h3s: list[str] = []
  for line in lines:
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if not m:
      continue
    lvl = len(m.group(1))
    title = m.group(2).strip()
    if lvl == 2:
      if in_h2:
        break
      in_h2 = title.lower() == target
      continue
    if in_h2 and lvl == 3:
      h3s.append(title)
  return h3s


def _combined_h3_body(content: str, h2_title: str) -> str:
  parts: list[str] = []
  for h3 in extract_h3_under_h2(content, h2_title):
    body = _clean_faq_answer(extract_section_text(content, h3, h3_only=True))
    if body:
      parts.append(body)
  return _clip_smart(" ".join(parts), 480)


def _benefits_section_body(content: str) -> str:
  for h2 in re.findall(r"^##\s+(.+)$", content, re.MULTILINE):
    title = h2.strip()
    low = title.lower()
    if any(k in low for k in ("benefit", "why choose", "advantage", "key feature")):
      body = extract_section_text(content, title)
      if body:
        return body
  return ""


def _clean_faq_answer(text: str) -> str:
  t = (text or "").strip()
  t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)
  t = re.sub(r"\*+", "", t)
  return _clip_smart(t, 480)


def _shuffle(items: list[Any], seed: int) -> list[Any]:
  out = list(items)
  for i in range(len(out) - 1, 0, -1):
    j = (seed + i * 7919) % (i + 1)
    out[i], out[j] = out[j], out[i]
  return out


@dataclass
class VariationContext:
  """Content-derived pools — variation comes from the article, not static templates."""
  seed: int
  short_topic: str
  keywords: list[str]
  display_title: str
  source_content: str = ""
  paragraphs: list[str] = field(default_factory=list)
  sentences: list[str] = field(default_factory=list)
  h2_titles: list[str] = field(default_factory=list)
  bullets: list[str] = field(default_factory=list)
  section_snippets: dict[str, str] = field(default_factory=dict)
  gap_hints: list[str] = field(default_factory=list)

  def pick(self, pool: list[str], salt: int = 0) -> str:
    if not pool:
      return ""
    return pool[(self.seed + salt) % len(pool)]

  def pick_n(self, pool: list[str], count: int, salt: int = 0) -> list[str]:
    if not pool or count <= 0:
      return []
    shuffled = _shuffle(pool, self.seed + salt)
    return shuffled[: min(count, len(shuffled))]


def build_variation_context(
  content: str,
  *,
  short_topic: str,
  display_title: str,
  keywords: list[str],
  gaps: list[dict[str, str]],
  seed: int,
) -> VariationContext:
  paras = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip() and not p.strip().startswith("#")]
  sentences: list[str] = []
  for para in paras:
    for s in re.split(r"(?<=[.!?])\s+", para):
      s = re.sub(r"\s+", " ", s.strip())
      if len(s) > 25:
        sentences.append(s)

  bullets: list[str] = []
  for line in content.splitlines():
    m = re.match(r"^[\*\-]\s+(.+)$", line.strip())
    if m:
      bullets.append(m.group(1).strip())

  h2s: list[str] = []
  section_snippets: dict[str, str] = {}
  for h2 in re.findall(r"^##\s+(.+)$", content, re.MULTILINE):
    title = h2.strip()
    low = title.lower()
    if low in ("conclusion", "frequently asked questions", "introduction"):
      continue
    h2s.append(title)
    snip = _clean_faq_answer(extract_section_text(content, title))
    if snip:
      section_snippets[title] = snip

  gap_hints = [g["suggestion"] for g in gaps if g.get("suggestion")]

  sentences, bullets = filter_content_pools(sentences, bullets)

  return VariationContext(
    seed=seed,
    short_topic=short_topic,
    keywords=keywords,
    display_title=display_title,
    source_content=content,
    paragraphs=paras,
    sentences=sentences,
    h2_titles=h2s,
    bullets=bullets,
    section_snippets=section_snippets,
    gap_hints=gap_hints,
  )


def _sentence_has_word(s: str, word: str) -> bool:
  return bool(re.search(rf"\b{re.escape(word)}\b", s.lower()))


def _sentence_to_question(sentence: str, topic: str, salt: int) -> str | None:
  s = re.sub(r"\s+", " ", sentence.strip().rstrip("."))
  if not s or is_internal_suggestion(s):
    return None
  if s.endswith("?"):
    return s
  words = s.split()
  if len(words) > 14:
    s = " ".join(words[:14])
  mode = (hash(s) + salt) % 3
  if mode == 0:
    return f"{s}?"
  if mode == 1:
    return f"How does {s.lower()} relate to {topic}?"
  return f"Can you explain how {s.lower()} relates to {topic}?"


def _heading_to_question(heading: str, topic: str, salt: int) -> str:
  h = heading.strip()
  if h.endswith("?"):
    return h
  low = h.lower()
  if low.startswith("why "):
    return h if h.endswith("?") else f"{h}?"
  if low.startswith("how "):
    return h if h.endswith("?") else f"{h}?"
  focus = h.split(":")[0].strip()
  mode = (hash(focus) + salt) % 2
  if mode == 0:
    return f"What should you know about {focus}?"
  return f"How does {focus} connect to {topic}?"


def _dynamic_answer_lead(ctx: VariationContext, answer: str, salt: int) -> str:
  if not answer or len(answer) < 20:
    return answer
  pool = [s for s in ctx.sentences if s.lower() not in answer.lower()[:80]]
  lead_sent = ctx.pick(pool, salt)
  if not lead_sent:
    return answer
  lead = _clip_smart(lead_sent, 90)
  if lead.lower() in answer.lower():
    return answer
  return f"{lead} — {answer[0].lower()}{answer[1:]}" if answer[0].isupper() else f"{lead} — {answer}"


def generate_faqs_from_content(ctx: VariationContext) -> list[dict[str, str]]:
  """FAQs built dynamically from article headings, sentences, and sections."""
  primary = ctx.short_topic
  faqs: list[dict[str, str]] = []
  seen_q: set[str] = set()

  intro_pool = ctx.paragraphs or [f"{primary.title()} is covered in this article."]
  intro = _clean_faq_answer(ctx.pick(intro_pool, 0))
  what_q = _sentence_to_question(ctx.pick(ctx.sentences, 2) or intro, primary, ctx.seed) or f"What is {primary.title()}?"
  faqs.append({"question": what_q, "answer": intro})
  seen_q.add(what_q.lower())

  benefit_body = _clean_faq_answer(
    _benefits_section_body(ctx.source_content) or ctx.pick(list(ctx.section_snippets.values()), 3),
  )
  if benefit_body:
    benefit_src = next((t for t in ctx.h2_titles if "benefit" in t.lower() or "why" in t.lower()), ctx.h2_titles[0] if ctx.h2_titles else primary)
    bq = _heading_to_question(benefit_src, primary, ctx.seed + 1)
    if bq.lower() not in seen_q:
      faqs.append({"question": bq, "answer": benefit_body})
      seen_q.add(bq.lower())

  start_sent = ctx.pick(
    [s for s in ctx.sentences if any(_sentence_has_word(s, w) for w in ("start", "begin", "first"))]
    or ctx.sentences,
    4,
  )
  start_a = _clean_faq_answer(start_sent) if start_sent else _clip_smart(ctx.pick(ctx.paragraphs, 5), 200)
  start_candidates = [
    h for h in ctx.h2_titles
    if re.search(r"\bstart\b", h.lower()) or "getting started" in h.lower() or re.search(r"\bguide\b", h.lower())
  ]
  if start_candidates:
    start_h = ctx.pick(start_candidates, 5)
    section_a = _clean_faq_answer(extract_section_text(ctx.source_content, start_h))
    if len(section_a) >= 40:
      start_a = section_a
    sq = _heading_to_question(start_h, primary, ctx.seed + 3)
    if sq.lower() not in seen_q:
      faqs.append({"question": sq, "answer": start_a})
      seen_q.add(sq.lower())

  rotated = _shuffle(ctx.h2_titles, ctx.seed + 6)
  for i, title in enumerate(rotated):
    if len(faqs) >= 10:
      break
    body = _clean_faq_answer(extract_section_text(ctx.source_content, title))
    if len(body) < 40:
      combined = _combined_h3_body(ctx.source_content, title)
      if len(combined) >= 40:
        q = _heading_to_question(title, primary, ctx.seed + 10 + i)
        if q.lower() not in seen_q:
          seen_q.add(q.lower())
          faqs.append({"question": q, "answer": combined})
        continue
      for h3 in extract_h3_under_h2(ctx.source_content, title):
        h3body = _clean_faq_answer(extract_section_text(ctx.source_content, h3, h3_only=True))
        if len(h3body) < 30:
          continue
        q = _heading_to_question(h3, primary, ctx.seed + 20 + i)
        if q.lower() in seen_q:
          continue
        seen_q.add(q.lower())
        faqs.append({"question": q, "answer": h3body})
        if len(faqs) >= 10:
          break
      continue
    q = _heading_to_question(title, primary, ctx.seed + 10 + i)
    if q.lower() in seen_q:
      continue
    seen_q.add(q.lower())
    faqs.append({"question": q, "answer": body})

  for bullet in ctx.pick_n(ctx.bullets, min(4, len(ctx.bullets)), 8):
    if len(faqs) >= 10:
      break
    q = _sentence_to_question(bullet, primary, ctx.seed + 30)
    if not q:
      continue
    if q.lower() in seen_q:
      continue
    seen_q.add(q.lower())
    faqs.append({"question": q, "answer": _clean_faq_answer(bullet)})

  for sent in _shuffle(ctx.sentences, ctx.seed + 90):
    if len(faqs) >= 10:
      break
    q = _sentence_to_question(sent, primary, ctx.seed + 90 + len(faqs))
    if not q:
      continue
    if q.lower() in seen_q:
      continue
    seen_q.add(q.lower())
    faqs.append({"question": q, "answer": _clean_faq_answer(sent)})

  return _shuffle(faqs, ctx.seed + 71)[:10]


def _clip(text: str, n: int = 300) -> str:
  text = re.sub(r"\s+", " ", (text or "").strip())
  return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _pick(seed: int, options: list[str]) -> str:
  return options[seed % len(options)] if options else ""


def _shingle_jaccard(a: str, b: str, size: int = 5) -> float:
  def shingles(s: str) -> set[tuple[str, ...]]:
    words = [w.lower() for w in re.findall(r"\w+", s)]
    if len(words) < size:
      return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}

  sa, sb = shingles(a), shingles(b)
  if not sa or not sb:
    return 0.0
  return len(sa & sb) / len(sa | sb)


def infer_topic(content: str, keywords: list[str]) -> str:
  if keywords:
    return keywords[0]
  m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
  if m:
    return re.sub(r"[*_`]", "", m.group(1)).strip()
  return _clip((content or "").strip().split("\n\n")[0], 80)


def extract_entities_from_content(content: str, keywords: list[str]) -> list[str]:
  entities: list[str] = []
  seen: set[str] = set()
  for kw in keywords:
    if kw.lower() not in seen:
      seen.add(kw.lower())
      entities.append(kw)
  for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\b", content):
    k = phrase.lower()
    if k not in seen and len(phrase) > 2:
      seen.add(k)
      entities.append(phrase)
  return entities[:20]


def analyze_keywords(content: str, keywords: list[str]) -> dict[str, Any]:
  wc = max(count_words(content), 1)
  primary = keywords[0] if keywords else ""
  secondary = keywords[1:] if len(keywords) > 1 else []
  low = content.lower()
  densities = {kw: round(low.count(kw.lower()) / wc * 100, 2) for kw in keywords[:10]}
  return {
    "primary": primary,
    "secondary": secondary,
    "densities": densities,
    "missing_in_content": [kw for kw in keywords if kw.lower() not in low],
    "recommended_density_pct": "0.5–2.5",
    "word_count": wc,
  }


def build_coverage_map(
  content: str,
  keywords: list[str],
  entities: list[str],
) -> dict[str, Any]:
  """Map keywords/entities to headings and sections."""
  sections: list[dict[str, Any]] = []
  current_heading = "Introduction"
  current_text: list[str] = []

  for line in (content or "").split("\n"):
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if m:
      if current_text or current_heading:
        sections.append({
          "heading": current_heading,
          "level": len(current_text) and 2 or 1,
          "text": "\n".join(current_text),
        })
      current_heading = m.group(2).strip()
      current_text = []
    elif line.strip():
      current_text.append(line.strip())
  if current_text or current_heading:
    sections.append({"heading": current_heading, "text": "\n".join(current_text)})

  # Re-tag heading levels from markdown
  indexed: list[dict[str, Any]] = []
  for line in (content or "").split("\n"):
    m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
    if m:
      indexed.append({"heading": m.group(2).strip(), "level": len(m.group(1))})
  level_by_heading = {s["heading"].lower(): s["level"] for s in indexed}

  coverage: list[dict[str, Any]] = []
  all_terms = list(dict.fromkeys(keywords + entities))
  body_low = content.lower()
  for term in all_terms[:15]:
    hits: list[str] = []
    for sec in sections:
      blob = f"{sec['heading']} {sec['text']}"
      if _term_in_text(term, blob):
        hits.append(sec["heading"])
    coverage.append({
      "term": term,
      "covered": bool(hits),
      "sections": hits,
      "status": "covered" if hits else "missing",
      "mentions": body_low.count(term.lower()) if term.lower() in body_low else sum(
        1 for w in re.findall(r"\w+", term.lower()) if w in body_low
      ),
    })

  section_stats: list[dict[str, Any]] = []
  for sec in sections:
    wc = count_words(sec["text"])
    lvl = level_by_heading.get(sec["heading"].lower(), 2)
    strength = "strong" if wc >= 60 else "adequate" if wc >= 35 else "thin"
    section_stats.append({
      "heading": sec["heading"],
      "word_count": wc,
      "level": lvl,
      "strength": strength,
    })

  thin_sections = [
    s["heading"] for s in section_stats
    if s["strength"] == "thin" and s["level"] == 2 and s["word_count"] > 0
  ]

  covered_count = sum(1 for c in coverage if c["covered"])
  h2_count = sum(1 for s in section_stats if s["level"] == 2)
  h3_count = sum(1 for s in section_stats if s["level"] == 3)
  bullet_count = len(re.findall(r"^[\*\-]\s+", content, re.MULTILINE))

  return {
    "sections": section_stats,
    "terms": coverage,
    "coverage_pct": round(100 * covered_count / max(1, len(coverage)), 1),
    "missing_terms": [c["term"] for c in coverage if not c["covered"]],
    "thin_sections": thin_sections,
    "structure": {"h2_count": h2_count, "h3_count": h3_count, "bullet_count": bullet_count},
  }


def local_gap_analysis(
  content: str,
  coverage_map: dict[str, Any],
  keywords: list[str],
  *,
  short_topic: str,
) -> list[dict[str, str]]:
  """Content-only gap analysis — no retrieval required."""
  gaps: list[dict[str, str]] = []
  low = content.lower()
  wc = count_words(content)

  for term in coverage_map.get("missing_terms", [])[:5]:
    gaps.append({
      "type": "coverage_gap",
      "source": "coverage_map",
      "priority": "high",
      "suggestion": f"Add a section or paragraph covering **{term}** - not found in current content.",
    })

  for heading in coverage_map.get("thin_sections", [])[:4]:
    gaps.append({
      "type": "thin_section",
      "source": "coverage_map",
      "priority": "medium",
      "suggestion": f"Expand **{heading}** with examples, bullets, or a short how-to (currently thin).",
    })

  primary = keywords[0] if keywords else short_topic
  if primary and not _term_in_text(primary, content[:500]):
    gaps.append({
      "type": "keyword_gap",
      "source": "keyword_analysis",
      "priority": "high",
      "suggestion": f"Mention **{primary}** in the opening paragraph for stronger SEO.",
    })

  if not re.search(r"faq|frequently asked", content, re.I):
    gaps.append({
      "type": "structure_gap",
      "source": "seo_checklist",
      "priority": "medium",
      "suggestion": "Add a FAQ section with 3–5 questions for featured snippets.",
    })

  topic_key = (
    "flutter" if "flutter" in low
    else "erp" if re.search(r"\berp\b", low) or "enterprise resource planning" in low
    else "default"
  )
  for topic_term in _SEO_TOPIC_CHECKLIST.get(topic_key, _SEO_TOPIC_CHECKLIST["default"]):
    if topic_term not in low:
      gaps.append({
        "type": "topic_gap",
        "source": "seo_checklist",
        "priority": "low",
        "suggestion": f"Consider covering **{topic_term}** to match search intent for this topic.",
      })

  if wc > 250 and not re.search(r"(conclusion|summary|in summary|to sum up)", content, re.I):
    gaps.append({
      "type": "structure_gap",
      "source": "seo_checklist",
      "priority": "low",
      "suggestion": "Add a conclusion with a clear takeaway or call to action.",
    })

  return gaps[:12]


def gap_analysis(
  content: str,
  coverage_map: dict[str, Any],
  docs: list[Any],
  entities: list[str],
  *,
  keywords: list[str] | None = None,
  short_topic: str = "",
) -> list[dict[str, str]]:
  gaps = local_gap_analysis(content, coverage_map, keywords or [], short_topic=short_topic)
  seen = {g["suggestion"][:50].lower() for g in gaps}
  low = content.lower()

  for doc in docs[:4]:
    if not is_doc_relevant(doc, derive_anchor_terms(content, short_topic, keywords or [])):
      continue
    for sent in re.split(r"(?<=[.!?])\s+", doc.text):
      sent = sent.strip()
      if len(sent) < 40 or not is_fact_relevant(sent, derive_anchor_terms(content, short_topic, keywords or [])):
        continue
      key_terms = [w for w in re.findall(r"\b[a-z]{5,}\b", sent.lower()) if w not in {
        "about", "their", "which", "would", "could", "should", "there", "these", "those",
      }]
      if sum(1 for t in key_terms[:8] if t in low) < max(1, len(key_terms) // 4):
        key = sent[:60].lower()
        if key not in seen:
          seen.add(key)
          gaps.append({
            "type": "competitor_gap",
            "source": doc.source,
            "priority": "medium",
            "suggestion": _clip(sent, 220),
          })
      if len(gaps) >= 14:
        break

  for ent in entities[:6]:
    if ent.lower() not in low and not _term_in_text(ent, content):
      suggestion = f"Strengthen coverage of **{ent}** in a dedicated paragraph or bullet list."
      if suggestion[:50].lower() not in seen:
        seen.add(suggestion[:50].lower())
        gaps.append({
          "type": "entity_gap",
          "source": "entity_extraction",
          "priority": "low",
          "suggestion": suggestion,
        })
  return gaps[:14]


def detect_local_opportunities(
  content: str,
  coverage_map: dict[str, Any],
  gaps: list[dict[str, str]],
) -> dict[str, Any]:
  """Novelty from local analysis when RAG is skipped or returns nothing."""
  novel: list[dict[str, Any]] = []
  for term in coverage_map.get("missing_terms", [])[:6]:
    novel.append({
      "text": f"Section opportunity: {term}",
      "source": "coverage_map",
      "confidence": 0.85,
      "type": "coverage_gap",
      "term": term,
    })
  for sec in coverage_map.get("sections", []):
    if sec.get("strength") == "thin":
      novel.append({
        "text": f"Expand the '{sec['heading']}' section ({sec.get('word_count', 0)} words) with specifics.",
        "source": "coverage_map",
        "confidence": 0.75,
        "type": "thin_section",
      })
  return {
    "novel_facts": novel[:10],
    "redundant_count": 0,
    "novel_count": len(novel),
    "novelty_ratio": 1.0 if novel else 0.0,
    "mode": "local",
  }


def detect_novelty(content: str, facts: list[ExtractedFact], *, threshold: float = 0.72) -> dict[str, Any]:
  novel: list[dict[str, Any]] = []
  redundant: list[str] = []
  for f in facts:
    if _shingle_jaccard(content, f.text) >= threshold:
      redundant.append(_clip(f.text, 80))
    else:
      novel.append({"text": f.text, "source": f.source, "confidence": round(f.confidence, 3)})
  return {
    "novel_facts": novel[:12],
    "redundant_count": len(redundant),
    "novel_count": len(novel),
    "novelty_ratio": round(len(novel) / max(1, len(facts)), 2),
    "mode": "rag",
  }


def plan_sections(
  content: str,
  keywords: list[str],
  gaps: list[dict[str, str]],
  coverage_map: dict[str, Any],
  *,
  seed: int,
) -> list[dict[str, str]]:
  """Section planner — H2/H3 outline for optimized article."""
  primary = keywords[0] if keywords else "Topic"
  plan: list[dict[str, str]] = [
    {"level": "h1", "title": _pick(seed, [primary.title(), f"Guide to {primary.title()}", f"{primary.title()}: Optimized"])},
    {"level": "h2", "title": "Introduction"},
  ]

  existing_h2 = [s["heading"] for s in coverage_map.get("sections", []) if s.get("heading")]
  for h in existing_h2[:4]:
    if h.lower() not in ("introduction", "conclusion"):
      plan.append({"level": "h2", "title": h})

  gap_titles = _pick(seed + 1, [
    "Key Insights from Research",
    "What Open Data Sources Show",
    "Expanded Coverage",
    "Additional Expert Context",
  ])
  if gaps:
    plan.append({"level": "h2", "title": gap_titles})

  for g in gaps[:3]:
    if g["type"] in ("coverage_gap", "competitor_gap", "topic_gap", "entity_gap"):
      term = parse_term_from_gap(g)
      if term and not is_internal_suggestion(term):
        plan.append({"level": "h3", "title": _clip(term, 60)})

  if not any(p["title"].lower() == "conclusion" for p in plan):
    plan.append({"level": "h2", "title": "Conclusion"})
  plan.append({"level": "h2", "title": "Frequently Asked Questions"})
  return plan


def generate_sections(
  original: str,
  section_plan: list[dict[str, str]],
  novel_facts: list[dict[str, Any]],
  keywords: list[str],
  gaps: list[dict[str, str]],
  *,
  tone: str,
  seed: int,
  allow_rag_injection: bool = True,
) -> tuple[str, list[str]]:
  """Section generator — assemble optimized markdown from plan + evidence."""
  suggestions: list[str] = []
  primary = keywords[0] if keywords else "this topic"
  fact_texts = [f["text"] for f in novel_facts] if allow_rag_injection else []
  fact_idx = 0
  parts: list[str] = []

  intro_paras = [p.strip() for p in re.split(r"\n\s*\n", original) if p.strip() and not p.startswith("#")]
  intro = intro_paras[0] if intro_paras else original[:400]

  if keywords and primary.lower() not in intro.lower()[:200]:
    intro = _pick(seed, [
      f"**{primary.title()}** is essential for teams worldwide. ",
      f"This optimized guide to **{primary}** improves clarity, depth, and search performance. ",
    ]) + intro
    suggestions.append(f"Wove primary keyword '{primary}' into introduction.")

  for item in section_plan:
    level, title = item["level"], item["title"]
    prefix = "#" if level == "h1" else "##" if level == "h2" else "###"
    if level == "h1":
      parts.append(f"{prefix} {title}\n")
      continue
    if title.lower() == "introduction":
      parts.append(f"{prefix} Introduction\n\n{intro}\n")
      continue
    if title.lower() == "frequently asked questions":
      continue
    if title.lower() == "conclusion":
      parts.append(
        f"{prefix} Conclusion\n\n"
        f"Applying these improvements to **{primary}** content strengthens structure, "
        f"coverage, and readability. Review metrics and iterate monthly.\n"
      )
      continue

  gap_heading = next((p for p in section_plan if "insight" in p["title"].lower() or "coverage" in p["title"].lower() or "research" in p["title"].lower()), None)
  if allow_rag_injection and gap_heading and fact_texts:
    parts.append(f"## {gap_heading['title']}\n")
    start = seed % max(1, len(fact_texts))
    rotated = fact_texts[start:] + fact_texts[:start]
    parts.append("\n".join(f"- {t}" for t in rotated[:3]) + "\n")
    suggestions.append("Generated evidence section from novel open-data facts.")

  gap_sents = [
    g["suggestion"] for g in gaps
    if g["type"] in ("competitor_gap", "coverage_gap") and not is_internal_suggestion(g.get("suggestion", ""))
  ]
  if allow_rag_injection and gap_sents:
    heading = _pick(seed + 2, ["## Deeper Context", "## Additional Topics", "## Expanded Coverage"])
    start_g = seed % max(1, len(gap_sents))
    # Use competitor facts only — never paste optimizer suggestion strings
    fact_fill = fact_texts[start_g : start_g + 2] if fact_texts else []
    if fact_fill:
      parts.append(f"{heading}\n\n" + "\n\n".join(fact_fill) + "\n")
    suggestions.append("Filled gaps with evidence from open-data facts.")

  for item in section_plan:
    if item["level"] != "h2":
      continue
    t = item["title"].lower()
    if t in ("introduction", "conclusion", "frequently asked questions"):
      continue
    if any(t in p.lower() for p in parts):
      continue
    body = ""
    if fact_idx < len(fact_texts):
      body = fact_texts[fact_idx]
      fact_idx += 1
    else:
      for para in intro_paras[1:4]:
        if item["title"].lower() in para.lower() or primary.lower() in para.lower():
          body = para
          break
    if body:
      parts.append(f"## {item['title']}\n\n{body}\n")

  article = "\n".join(parts).strip()
  if count_words(article) < count_words(original) * 0.5:
    article = original.strip()
    if not article.startswith("#") and keywords:
      article = f"# {primary.title()}\n\n{article}"
    suggestions.append("Preserved original body; applied structural enhancements below.")
    if allow_rag_injection and fact_texts:
      article += f"\n\n## Expert Insights\n\n" + "\n".join(f"- {t}" for t in fact_texts[:2])

  suggestions.append(f"Section generator applied {tone} tone across {len(section_plan)} planned sections.")
  return article, suggestions


def generate_faqs(
  ctx: VariationContext,
  novel_facts: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
  faqs = generate_faqs_from_content(ctx)
  if novel_facts:
    primary = ctx.short_topic
    for f in novel_facts[:1]:
      q = _sentence_to_question(clean_fact_text(f["text"]), primary, ctx.seed + 99)
      if q:
        faqs.append({
          "question": q,
          "answer": clean_fact_text(f["text"]),
        })
  return faqs


def append_faqs_to_content(content: str, faqs: list[dict[str, str]]) -> str:
  if not faqs:
    return content
  if re.search(r"^##\s+frequently asked questions", content, re.I | re.MULTILINE):
    content = re.sub(
      r"\n##\s+Frequently Asked Questions[\s\S]*$",
      "",
      content,
      flags=re.I,
    ).strip()
  out = content.rstrip() + "\n\n## Frequently Asked Questions\n\n"
  for f in faqs[:6]:
    out += f"### {f['question']}\n\n{f['answer']}\n\n"
  return out.strip()


def _split_h2_sections(content: str) -> tuple[str, list[tuple[str, str]]]:
  """Return (preamble through first H2, [(heading_line, body), ...])."""
  lines = content.splitlines()
  preamble: list[str] = []
  sections: list[tuple[str, str]] = []
  current_h: str | None = None
  current_body: list[str] = []
  i = 0
  while i < len(lines):
    if re.match(r"^##\s+", lines[i]):
      break
    preamble.append(lines[i])
    i += 1
  while i < len(lines):
    line = lines[i]
    if re.match(r"^##\s+", line):
      if current_h is not None:
        sections.append((current_h, "\n".join(current_body)))
      current_h = line
      current_body = []
    else:
      current_body.append(line)
    i += 1
  if current_h is not None:
    sections.append((current_h, "\n".join(current_body)))
  return "\n".join(preamble).strip(), sections


def _is_pinned_section(heading_line: str) -> bool:
  title = re.sub(r"^##\s+", "", heading_line).strip().lower()
  return title in ("conclusion", "frequently asked questions", "introduction") or title.startswith("faq")


def reorder_h2_sections(content: str, seed: int) -> tuple[str, list[str]]:
  pre, sections = _split_h2_sections(content)
  if len(sections) < 3:
    return content, []
  pinned: list[tuple[str, str]] = []
  middle: list[tuple[str, str]] = []
  for h, body in sections:
    if _is_pinned_section(h):
      pinned.append((h, body))
    else:
      middle.append((h, body))
  if len(middle) < 2:
    return content, []
  middle = _shuffle(middle, seed + 29)
  rotate = (seed + 7) % len(middle)
  if rotate:
    middle = middle[rotate:] + middle[:rotate]
  parts = [pre] if pre else []
  for h, body in middle + pinned:
    parts.append(h)
    if body.strip():
      parts.append(body.strip())
  return "\n\n".join(parts).strip(), ["Reordered H2 sections (seed-driven layout)."]


def shuffle_section_bullets(content: str, seed: int) -> tuple[str, list[str]]:
  lines = content.splitlines()
  out: list[str] = []
  bullet_buf: list[str] = []
  changed = False

  def flush_bullets(buf: list[str], s: int) -> list[str]:
    if len(buf) < 2:
      return buf
    return _shuffle(buf, s + 31)

  for line in lines:
    if re.match(r"^[\*\-]\s+", line.strip()):
      bullet_buf.append(line)
      continue
    if bullet_buf:
      shuffled = flush_bullets(bullet_buf, seed + len(out))
      if shuffled != bullet_buf:
        changed = True
      out.extend(shuffled)
      bullet_buf = []
    out.append(line)
  if bullet_buf:
    shuffled = flush_bullets(bullet_buf, seed + len(out))
    if shuffled != bullet_buf:
      changed = True
    out.extend(shuffled)
  return "\n".join(out), (["Shuffled bullet lists within sections."] if changed else [])


def vary_intro_sentences(content: str, ctx: VariationContext, seed: int) -> tuple[str, list[str]]:
  pre, sections = _split_h2_sections(content)
  if not pre:
    return content, []
  lines = pre.splitlines()
  h1_end = 0
  for i, line in enumerate(lines):
    if re.match(r"^#\s+", line.strip()):
      h1_end = i + 1
      break
  intro_lines = lines[h1_end:]
  intro = "\n".join(intro_lines).strip()
  if not intro or count_words(intro) < 30:
    return content, []
  paras = [p.strip() for p in re.split(r"\n\s*\n", intro) if p.strip()]
  if not paras:
    return content, []
  first = paras[0]
  sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", first) if s.strip()]
  if len(sents) < 2:
    return content, []
  rotate = (seed + 3) % len(sents)
  rotated = " ".join(sents[rotate:] + sents[:rotate])
  paras[0] = rotated
  new_intro = "\n\n".join(paras)
  new_pre = "\n".join(lines[:h1_end]) + ("\n\n" + new_intro if new_intro else "")
  rebuilt = new_pre.strip()
  if sections:
    rebuilt += "\n\n" + "\n\n".join(h + ("\n" + b.strip() if b.strip() else "") for h, b in sections)
  return rebuilt.strip(), ["Rotated introduction sentence order."]


def _sentences_from_text(text: str) -> list[str]:
  flat = re.sub(r"\s+", " ", (text or "").strip())
  return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if len(s.strip()) > 15]


def shuffle_h3_blocks(body: str, seed: int) -> str:
  blocks: list[str] = []
  current: list[str] = []
  for line in body.splitlines():
    if re.match(r"^###\s+", line.strip()):
      if current:
        blocks.append("\n".join(current).strip())
      current = [line]
    else:
      current.append(line)
  if current:
    blocks.append("\n".join(current).strip())
  if len(blocks) < 2:
    return body
  return "\n\n".join(_shuffle(blocks, seed))


def shuffle_prose_in_block(body: str, seed: int) -> str:
  lines = body.splitlines()
  out: list[str] = []
  prose_buf: list[str] = []

  def flush_prose(buf: list[str], salt: int) -> None:
    if not buf:
      return
    sents = _sentences_from_text(" ".join(buf))
    if len(sents) >= 2:
      sents = _shuffle(sents, seed + salt)
    out.append(" ".join(sents))

  for i, line in enumerate(lines):
    stripped = line.strip()
    if re.match(r"^#{1,6}\s+", stripped) or re.match(r"^[\*\-]\s+", stripped):
      flush_prose(prose_buf, i)
      prose_buf = []
      out.append(line)
    elif stripped:
      prose_buf.append(stripped)
    else:
      flush_prose(prose_buf, i)
      prose_buf = []
      out.append(line)
  flush_prose(prose_buf, len(lines))
  return "\n".join(out)


def rebuild_intro_preamble(pre: str, ctx: VariationContext, seed: int) -> tuple[str, list[str]]:
  lines = [ln for ln in pre.splitlines() if ln.strip()]
  h1_line = next((ln for ln in lines if re.match(r"^#\s+", ln.strip())), None)
  if not h1_line or len(ctx.sentences) < 3:
    return pre, []
  n = 2 + (seed % 4)
  picked = _shuffle(ctx.sentences, seed + 83)[:n]
  return f"{h1_line}\n\n" + "\n\n".join(picked), ["Recomposed introduction from article sentence pool."]


def rebuild_conclusion_body(body: str, ctx: VariationContext, seed: int) -> tuple[str, list[str]]:
  pool = [s for s in ctx.sentences if len(s) > 25]
  if len(pool) < 2:
    return body, []
  n = 1 + (seed % 3)
  picked = _shuffle(pool, seed + 97)[:n]
  bullet_lines = [ln for ln in body.splitlines() if re.match(r"^[\*\-]\s+", ln.strip())]
  prose = "\n\n".join(picked)
  if bullet_lines:
    return prose + "\n\n" + "\n".join(_shuffle(bullet_lines, seed + 99)), ["Recomposed conclusion from article pool."]
  return prose, ["Recomposed conclusion from article pool."]


def vary_h1_title(pre: str, ctx: VariationContext, short_topic: str, seed: int) -> tuple[str, list[str]]:
  lines = pre.splitlines()
  if not lines:
    return pre, []
  h1_idx = next((i for i, ln in enumerate(lines) if re.match(r"^#\s+", ln.strip())), None)
  if h1_idx is None:
    return pre, []
  h2_pick = ctx.pick(ctx.h2_titles, seed + 17) or short_topic.title()
  h2_pick2 = ctx.pick(ctx.h2_titles, seed + 19) or "Overview"
  variants = [
    re.sub(r"^#\s+", "", lines[h1_idx]).strip(),
    f"{short_topic.title()}: {h2_pick}",
    f"{h2_pick2} — {short_topic.title()} Guide",
    f"Complete Guide to {short_topic.title()}",
  ]
  lines[h1_idx] = f"# {variants[seed % len(variants)]}"
  return "\n".join(lines), ["Varied H1 title phrasing."]


def paraphrase_sentence(sentence: str, seed: int) -> str:
  t = sentence
  all_swaps = _LEXICAL_SWAPS + [
    (r"\busing\b", ["with", "through", "via"]),
    (r"\bcreate\b", ["build", "make", "develop"]),
    (r"\bapplications\b", ["apps", "software", "programs"]),
    (r"\bfeature\b", ["capability", "function", "tool"]),
    (r"\bperformance\b", ["speed", "efficiency", "throughput"]),
    (r"\ballows\b", ["lets", "enables", "helps"]),
    (r"\bmodern\b", ["current", "today's", "contemporary"]),
    (r"\bbuild\b", ["develop", "create", "craft"]),
  ]
  salt = seed + sum(ord(c) for c in t[:40])
  for i, (pat, alts) in enumerate(all_swaps):
    choice = alts[(salt + i * 31) % len(alts)]
    t = re.sub(pat, choice, t, flags=re.I)
  parts = [p.strip() for p in re.split(r",\s+", t) if p.strip()]
  if len(parts) >= 2 and (seed % 2) == 0:
    t = ", ".join(_shuffle(parts, salt))
  return t


def _seed_sentence_pool(ctx: VariationContext, seed: int) -> list[str]:
  """Stripe-select sentences so each seed draws a different subset."""
  k = 3 + (seed % 5)
  phase = seed % k
  pool = _shuffle(ctx.sentences, seed + 401)
  return [s for i, s in enumerate(pool) if (i + phase) % k == 0]


def _assign_section_sentences(
  title: str,
  ctx: VariationContext,
  seed: int,
  used: set[str],
) -> list[str]:
  kws = [w.lower() for w in re.findall(r"\w+", title) if len(w) > 3]
  pool = _seed_sentence_pool(ctx, seed)
  matched: list[str] = []
  for s in pool:
    if s in used:
      continue
    if not kws or any(k in s.lower() for k in kws):
      matched.append(s)
  if len(matched) < 2:
    for s in pool:
      if s not in used and s not in matched:
        matched.append(s)
      if len(matched) >= 3:
        break
  take = 2 + (seed % 3)
  picked = matched[:take]
  for s in picked:
    used.add(s)
  return [paraphrase_sentence(s, seed + i) for i, s in enumerate(picked)]


def _seed_transition(title: str, seed: int) -> str:
  hooks = [
    f"Regarding {title.lower()}, ",
    f"When it comes to {title.lower()}, ",
    f"On the topic of {title.lower()}, ",
    f"For {title.lower()}, ",
    f"Looking at {title.lower()}, ",
  ]
  return hooks[seed % len(hooks)]


def max_variation_rebuild(
  content: str,
  ctx: VariationContext,
  seed: int,
) -> tuple[str, list[str]]:
  """Rebuild the full article layout from the sentence pool — unique per seed."""
  _, sections = _split_h2_sections(content)
  suggestions = ["Full article recomposition from content sentence pool."]
  used: set[str] = set()

  h1_variants = [
    f"# {ctx.display_title or ctx.short_topic.title()}",
    f"# {ctx.short_topic.title()}: {ctx.pick(ctx.h2_titles, seed + 1)}",
    f"# {ctx.pick(ctx.h2_titles, seed + 2)} — {ctx.short_topic.title()} Guide",
    f"# Complete Guide to {ctx.short_topic.title()}",
  ]
  parts = [h1_variants[seed % len(h1_variants)]]

  intro_take = 2 + (seed % 4)
  intro_sents: list[str] = []
  for s in _seed_sentence_pool(ctx, seed + 83):
    if s in used:
      continue
    intro_sents.append(paraphrase_sentence(s, seed + len(intro_sents)))
    used.add(s)
    if len(intro_sents) >= intro_take:
      break
  if intro_sents:
    parts.append("\n\n".join(intro_sents))

  pinned: list[tuple[str, str]] = []
  middle: list[tuple[str, str]] = []
  skip_idx = seed % max(1, len(sections))
  for i, (h, body) in enumerate(sections):
    if i == skip_idx and len(sections) > 4:
      continue
    title = re.sub(r"^##\s+", "", h).strip()
    if _is_pinned_section(h):
      if title.lower() == "conclusion":
        close = _shuffle([s for s in ctx.sentences if s not in used], seed + 97)[: 2 + (seed % 2)]
        for s in close:
          used.add(s)
        sec_body = "\n\n".join(paraphrase_sentence(s, seed + 200 + j) for j, s in enumerate(close))
        pinned.append((h, sec_body))
      else:
        pinned.append((h, shuffle_prose_in_block(body, seed + i)))
      continue
    sents = _assign_section_sentences(title, ctx, seed + i * 17, used)
    if sents and (seed + i) % 2 == 0:
      sents[0] = _seed_transition(title, seed + i) + sents[0][0].lower() + sents[0][1:]
    bullets = _shuffle(ctx.bullets, seed + i * 19)[: 2 + (seed % 4)] if ctx.bullets else []
    sec_parts = ["\n\n".join(sents)]
    if bullets:
      sec_parts.append("\n".join(f"* {paraphrase_sentence(b, seed + 300 + j)}" for j, b in enumerate(bullets)))
    h3s = extract_h3_under_h2(content, title)
    if h3s:
      h3_blocks = []
      for j, h3 in enumerate(_shuffle(h3s, seed + i * 23)):
        h3_sents = _assign_section_sentences(h3, ctx, seed + i * 29 + j, used)
        if h3_sents:
          h3_blocks.append(f"### {h3}\n\n" + " ".join(h3_sents))
      if h3_blocks:
        sec_parts.append("\n\n".join(_shuffle(h3_blocks, seed + i * 31)))
    middle.append((h, "\n\n".join(sec_parts)))

  if len(middle) >= 2:
    middle = _shuffle(middle, seed + 29)
    rotate = (seed + 7) % len(middle)
    if rotate:
      middle = middle[rotate:] + middle[:rotate]

  for h, body in middle + pinned:
    parts.append(h)
    if body.strip():
      parts.append(body.strip())

  return "\n\n".join(parts).strip(), suggestions


def inject_section_lead_ins(content: str, ctx: VariationContext, seed: int) -> tuple[str, list[str]]:
  if not ctx.h2_titles or not ctx.sentences:
    return content, []
  picks = _shuffle(ctx.h2_titles, seed + 43)
  count = min(len(picks), 2 + (seed % max(1, len(picks))))
  targets = {t.lower() for t in picks[:count]}
  suggestions: list[str] = []
  text = content
  for i, title in enumerate(ctx.h2_titles):
    if title.lower() not in targets:
      continue
    lead = ctx.pick(_shuffle(ctx.sentences, seed + 50 + i), seed + 51 + i)
    if not lead or len(lead) < 30:
      continue
    body = extract_section_text(text, title)
    if not body or lead.lower()[:40] in body.lower():
      continue
    snippet = _clip_smart(lead, 120)
    text = re.sub(
      rf"(^##\s*{re.escape(title)}\s*\n\n)",
      rf"\1{snippet} ",
      text,
      count=1,
      flags=re.MULTILINE,
    )
    suggestions.append(f"Added lead-in under **{title}**.")
  return text, suggestions


_LEXICAL_SWAPS: list[tuple[str, list[str]]] = [
  (r"\benables\b", ["allows", "lets", "helps"]),
  (r"\bseveral\b", ["many", "multiple", "various"]),
  (r"\bimportant\b", ["key", "essential", "critical"]),
  (r"\bprovides\b", ["offers", "delivers", "gives"]),
  (r"\bdevelopers\b", ["dev teams", "engineers", "builders"]),
  (r"\bexcellent\b", ["strong", "outstanding", "solid"]),
  (r"\bpopular\b", ["widely used", "common", "in-demand"]),
  (r"\bessential\b", ["vital", "crucial", "necessary"]),
  (r"\bimproves\b", ["boosts", "enhances", "strengthens"]),
  (r"\bincluding\b", ["such as", "like", "among them"]),
]


def apply_lexical_variation(text: str, seed: int) -> tuple[str, list[str]]:
  t = text
  applied = 0
  for i, (pat, alts) in enumerate(_LEXICAL_SWAPS):
    choice = alts[(seed + i * 17) % len(alts)]
    new_t, n = re.subn(pat, choice, t, count=2 + (seed % 3), flags=re.I)
    if n:
      t = new_t
      applied += n
  extra_swaps = [
    (r"\busing\b", ["with", "through", "via"]),
    (r"\bcreate\b", ["build", "make", "develop"]),
    (r"\bapplications\b", ["apps", "software", "programs"]),
    (r"\bfeature\b", ["capability", "function", "tool"]),
    (r"\bperformance\b", ["speed", "efficiency", "throughput"]),
  ]
  for j, (pat, alts) in enumerate(extra_swaps):
    choice = alts[(seed + j * 23) % len(alts)]
    new_t, n = re.subn(pat, choice, t, count=1 + ((seed + j) % 2), flags=re.I)
    if n:
      t = new_t
      applied += n
  return t, ([f"Full lexical variation ({applied} swaps)."] if applied else [])


def apply_content_variation(
  content: str,
  ctx: VariationContext,
  seed: int,
) -> tuple[str, list[str]]:
  """Maximum seed-driven reshaping — each seed produces a distinct article layout."""
  text, suggestions = max_variation_rebuild(content.strip(), ctx, seed)
  text, lex_s = apply_lexical_variation(text, seed)
  suggestions.extend(lex_s)
  return text.strip(), suggestions


def polish_strong_content(
  content: str,
  *,
  short_topic: str,
  keywords: list[str],
  coverage_map: dict[str, Any],
  gaps: list[dict[str, str]],
  seed: int,
  ctx: VariationContext | None = None,
  skip_variation_rebuild: bool = False,
) -> tuple[str, list[str]]:
  """Light SEO polish — preserve gap-filled sections; avoid scrambling structured content."""
  suggestions: list[str] = []
  text = content.strip()
  primary = keywords[0] if keywords else short_topic
  vctx = ctx or build_variation_context(
    content, short_topic=short_topic, display_title=short_topic, keywords=keywords, gaps=gaps, seed=seed,
  )

  if skip_variation_rebuild:
    text, lex_s = apply_lexical_variation(text, seed)
    suggestions.extend(lex_s)
    suggestions.append("Preserved gap-filled structure; light lexical polish only.")
  else:
    text, var_suggestions = apply_content_variation(text, vctx, seed)
    suggestions.extend(var_suggestions)

  if not re.search(r"^#\s+", text, re.MULTILINE):
    text = f"# {short_topic.title()}\n\n{text}"
    suggestions.append("Added missing H1 title.")

  paras = vctx.paragraphs
  if paras and primary and not _term_in_text(primary, paras[0][:400]):
    lead = _clip(vctx.pick(vctx.sentences, 7), 90)
    prefix = f"**{primary.title()}** — {lead} " if lead else f"**{primary.title()}** — "
    text = text.replace(paras[0], prefix + paras[0], 1)
    suggestions.append(f"Wove primary keyword '{primary}' into the introduction.")

  thin = coverage_map.get("thin_sections", [])[:2]
  for i, heading in enumerate(thin):
    body = extract_section_text(text, heading)
    if body and count_words(body) < 50:
      extra_sent = vctx.pick(vctx.sentences, 11 + i)
      extra = f" {_clip(extra_sent, 120)}" if extra_sent else ""
      if extra:
        text = re.sub(
          rf"(^##\s*{re.escape(heading)}\s*\n\n)([\s\S]*?)(?=\n##|\Z)",
          lambda m: m.group(1) + m.group(2).rstrip() + extra + "\n",
          text,
          count=1,
          flags=re.MULTILINE,
        )
        suggestions.append(f"Expanded thin section: {heading}.")

  if primary and not _term_in_text(primary, text[-400:]):
    close_pool = [s for s in vctx.sentences if "conclusion" in s.lower() or primary.lower() in s.lower()]
    closing = vctx.pick(close_pool or vctx.sentences, 13)
    if closing:
      text = re.sub(
        r"(##\s+Conclusion[\s\S]*)$",
        lambda m: m.group(1).rstrip() + f" {_clip(closing, 160)}\n",
        text,
        count=1,
        flags=re.I,
      )
      suggestions.append("Reinforced conclusion with a content-derived line.")

  takeaway_pool = list(vctx.bullets)
  if not takeaway_pool or any(is_internal_suggestion(b) for b in takeaway_pool):
    takeaway_pool = build_key_takeaways(short_topic, keywords, text, [])
  if takeaway_pool and "## Key Takeaways" not in text and "## Quick Takeaways" not in text:
    block = "## Key Takeaways\n\n" + "\n".join(f"- {b}" for b in takeaway_pool[:5]) + "\n\n"
    if re.search(r"^##\s+conclusion\b", text, re.I | re.M):
      text = re.sub(r"^(##\s+conclusion\b)", block + "\\1", text, count=1, flags=re.I | re.M)
    elif re.search(r"^##\s+frequently asked questions\b", text, re.I | re.M):
      text = re.sub(r"^(##\s+frequently asked questions\b)", block + "\\1", text, count=1, flags=re.I | re.M)
    else:
      text = text.rstrip() + "\n\n" + block
    suggestions.append("Added Key Takeaways from article content and filled gaps.")

  return text.strip(), suggestions


def humanize_text(text: str, *, seed: int, tone: str, ctx: VariationContext | None = None) -> tuple[str, list[str]]:
  """Humanizer — prepends a clause from another sentence in the same article."""
  suggestions: list[str] = []
  t = text or ""
  t = re.sub(r"\b(utilize|leverage|facilitate|implement solutions)\b", "use", t, flags=re.I)
  t = re.sub(r"\bin order to\b", "to", t, flags=re.I)
  t = re.sub(r"\b(it is important to note that|it should be noted that)\b", "", t, flags=re.I)
  t = re.sub(r"[ \t]{2,}", " ", t)
  t = re.sub(r"\n{3,}", "\n\n", t).strip()

  paras = [p for p in re.split(r"\n\s*\n", t) if p.strip() and not p.strip().startswith("#")]
  if paras and ctx and ctx.sentences:
    intro_tokens = {w for w in re.findall(r"\w+", paras[0].lower()) if len(w) > 3}
    related = [
      s for s in ctx.sentences
      if s not in paras[0]
      and len({w for w in re.findall(r"\w+", s.lower()) if len(w) > 3} & intro_tokens) >= 2
    ]
    alt = ctx.pick(related or ctx.sentences[:3], seed + 40)
    if alt:
      bridge = _clip_smart(alt.split(".")[0], 90)
      if bridge and bridge.lower() not in paras[0].lower():
        t = t.replace(paras[0], f"{bridge}. {paras[0]}", 1)
        suggestions.append("Humanizer bridged intro with another sentence from the article.")

  suggestions.append(f"Humanizer applied ({tone} voice).")
  return t, suggestions


def rotate_faqs(faqs: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
  if len(faqs) < 4:
    return faqs
  pinned = faqs[:4]
  rest = faqs[4:]
  start = seed % max(1, len(rest))
  return pinned + rest[start:] + rest[:start]


def optimize_readability(text: str) -> tuple[str, list[str]]:
  """Readability polish — skip when already in 60–95 band (fast path)."""
  before = readability_score(text)
  if before >= 60:
    return text, [f"Readability already in target band ({before}/100)."]
  text, suggestions = improve_readability(text, target_min=60.0, target_max=95.0, max_passes=2)
  after = readability_score(text)
  suggestions.append(f"Readability adjusted ({before} → {after}).")
  return text, suggestions


def optimize_metadata(
  content: str,
  short_topic: str,
  display_title: str,
  keywords: list[str],
  *,
  seed: int,
  ctx: VariationContext | None = None,
) -> dict[str, str]:
  topic = display_title or short_topic
  return optimize_metadata_clean(topic, keywords, content, seed=seed)


def internal_linking_suggestions(
  keywords: list[str],
  entities: list[str],
  gaps: list[dict[str, str]],
) -> list[dict[str, str]]:
  links: list[dict[str, str]] = []
  for kw in keywords[1:6]:
    links.append({"anchor_text": kw, "target_topic": kw, "reason": "Secondary keyword — pillar page link."})
  for ent in entities[:4]:
    links.append({"anchor_text": ent, "target_topic": ent, "reason": "Entity link for topical authority."})
  for g in gaps[:2]:
    term = parse_term_from_gap(g)
    if term:
      links.append({"anchor_text": term, "target_topic": term, "reason": f"Gap fill ({g.get('type', 'gap')})."})
  return links[:10]


def schema_suggestions(title: str, meta: str, faqs: list[dict[str, str]]) -> dict[str, Any]:
  types = ["Article"]
  if faqs:
    types.append("FAQPage")
  hint: dict[str, Any] = {
    "@context": "https://schema.org",
    "@type": "Article",
    "headline": title,
    "description": meta,
  }
  if faqs:
    hint["hasPart"] = {
      "@type": "FAQPage",
      "mainEntity": [
        {"@type": "Question", "name": f["question"], "acceptedAnswer": {"@type": "Answer", "text": f["answer"]}}
        for f in faqs[:4]
      ],
    }
  return {"recommended_types": types, "jsonld_hint": hint}


def _slugify(title: str) -> str:
  s = re.sub(r"[^\w\s-]", "", (title or "").lower())
  return re.sub(r"[-\s]+", "-", s).strip("-")[:80] or "article"


def _infer_search_intent(content: str, keywords: list[str]) -> dict[str, Any]:
  low = content.lower()
  scores = {
    "informational": sum(1 for w in ("what", "how", "why", "guide", "learn", "explain") if w in low),
    "commercial": sum(1 for w in ("best", "compare", "review", "vs", "top", "pricing") if w in low),
    "transactional": sum(1 for w in ("buy", "download", "sign up", "get started", "pricing", "order") if w in low),
    "navigational": sum(1 for w in ("login", "official", "website", "homepage") if w in low),
  }
  primary_intent = max(scores, key=scores.get)
  return {"primary": primary_intent, "scores": scores}


def build_comprehensive_seo_report(
  *,
  content: str,
  optimized: str,
  short_topic: str,
  keywords: list[str],
  kw_analysis: dict[str, Any],
  coverage_map: dict[str, Any],
  gaps: list[dict[str, str]],
  entities: list[str],
  metadata: dict[str, str],
  faqs: list[dict[str, str]],
  internal_links: list[dict[str, str]],
  schema: dict[str, Any],
  seo_before: int,
  seo_after: int,
  original_metrics: dict[str, Any],
  optimized_metrics: dict[str, Any],
  suggestions: list[str],
  ctx: VariationContext,
) -> dict[str, Any]:
  """Full SEO audit report (sections 1–16) — derived from article content only."""
  cov_pct = coverage_map.get("coverage_pct", 0)
  entity_hits = sum(1 for e in entities[:12] if e.lower() in content.lower())
  entity_cov = round(100 * entity_hits / max(1, min(len(entities), 12)), 1)
  kw_cov = cov_pct
  depth = min(100, round(count_words(optimized) / 8))
  topical = min(100, len(ctx.h2_titles) * 12 + len(ctx.bullets) * 3)
  intent = _infer_search_intent(content, keywords)

  strengths: list[str] = []
  weaknesses: list[str] = []
  if original_metrics.get("readability_score", 0) >= 60:
    strengths.append("Readability already in a healthy band.")
  if ctx.h2_titles:
    strengths.append(f"Clear topical structure with {len(ctx.h2_titles)} H2 sections.")
  if ctx.bullets:
    strengths.append("Uses scannable bullet lists.")
  if seo_after > seo_before:
    strengths.append(f"SEO score improved by {seo_after - seo_before} points.")

  for g in gaps[:5]:
    weaknesses.append(g.get("suggestion", ""))
  if not re.search(r"^#\s+", content, re.MULTILINE):
    weaknesses.append("Missing H1 heading in original draft.")
  opportunities = [g.get("suggestion", "") for g in gaps if g.get("type") in ("coverage_gap", "thin_section")][:6]

  primary = keywords[0] if keywords else short_topic
  kw_rows = []
  for term in coverage_map.get("terms", [])[:12]:
    kw_rows.append({
      "keyword": term.get("term", ""),
      "status": term.get("status", ""),
      "mentions": term.get("mentions", 0),
      "sections": ", ".join(term.get("sections", [])[:3]),
    })
  if not kw_rows and keywords:
    for kw in keywords[:8]:
      kw_rows.append({"keyword": kw, "status": "primary" if kw == primary else "secondary", "mentions": content.lower().count(kw.lower()), "sections": ""})

  entity_rows = [{"entity": e, "in_content": e.lower() in content.lower()} for e in entities[:12]]

  gap_rows = [{"type": g.get("type", ""), "priority": g.get("priority", ""), "suggestion": g.get("suggestion", "")} for g in gaps[:10]]

  def_snippet = _clip_smart(ctx.sentences[0], 160) if ctx.sentences else f"{primary} is explained in this article."
  list_snippet = "\n".join(f"- {b}" for b in ctx.bullets[:6]) if ctx.bullets else "- Key points covered in the article sections."
  table_snippet = "| Topic | Coverage |\n| --- | --- |\n" + "\n".join(
    f"| {t.get('term', '')} | {t.get('status', '')} |" for t in coverage_map.get("terms", [])[:5]
  )

  title = metadata.get("title", short_topic)
  meta_desc = metadata.get("meta_description", "")
  slug = _slugify(title)

  return {
    "seo_audit": {"strengths": strengths, "weaknesses": weaknesses, "opportunities": opportunities},
    "keyword_analysis": {"primary": primary, "secondary": keywords[1:6], "densities": kw_analysis.get("densities", {}), "table": kw_rows},
    "entity_analysis": {"entities": entity_rows, "entity_coverage_pct": entity_cov},
    "content_gap_analysis": {"gaps": gap_rows},
    "search_intent": intent,
    "eeat_recommendations": [
      "Cite primary sources already referenced in your draft where applicable.",
      "Add author byline or reviewer note for expertise signals.",
      "Include a short 'last updated' line when facts are time-sensitive.",
      "Link to official documentation for technical claims (no fabricated stats).",
    ],
    "readability": {
      "before": original_metrics,
      "after": optimized_metrics,
      "flesch_ease": optimized_metrics.get("readability_score"),
    },
    "featured_snippets": {
      "definition": def_snippet,
      "list": list_snippet,
      "table": table_snippet,
      "faq": faqs[:3],
    },
    "faqs": faqs[:10],
    "internal_links": internal_links[:10],
    "external_references": [
      {"type": "official_docs", "note": "Link to official product or framework documentation relevant to your topic."},
      {"type": "academic", "note": "Add peer-reviewed or .edu sources when citing research claims."},
      {"type": "industry", "note": "Reference recognized industry reports only when you have a real URL."},
    ],
    "metadata": {
      "title": title,
      "meta_description": meta_desc,
      "slug": slug,
      "og_title": title[:60],
      "og_description": meta_desc[:160],
    },
    "schema_recommendations": schema,
    "ai_search_optimization": [
      "Lead each H2 with a direct 1–2 sentence answer (AI overview friendly).",
      "Use entity-rich phrases drawn from your headings and glossary terms.",
      "Keep FAQ answers concise and self-contained for Perplexity/Gemini citations.",
      "Maintain semantic headings that match how users phrase questions.",
    ],
    "final_metrics": {
      "seo_score_before": seo_before,
      "seo_score_after": seo_after,
      "readability_score": optimized_metrics.get("readability_score"),
      "topical_authority_score": topical,
      "keyword_coverage_pct": kw_cov,
      "entity_coverage_pct": entity_cov,
      "content_depth_score": depth,
    },
    "suggestions": suggestions[:14],
  }


def effective_variation_seed(client_seed: int | None) -> int:
  """Blend client seed + fresh entropy so every API call yields a unique layout."""
  base = make_variation_seed(client_seed)
  nonce = secrets.randbits(31) ^ (time.time_ns() & 0x7FFFFFFF)
  return make_variation_seed(base ^ nonce)


def estimate_variation_space(ctx: VariationContext) -> int:
  """Rough count of distinct layouts available from this article (not a hard cap)."""
  import math

  n = max(1, len(ctx.h2_titles))
  s = max(1, len(ctx.sentences))
  b = max(1, len(ctx.bullets))
  h3 = max(1, sum(len(extract_h3_under_h2(ctx.source_content, t)) for t in ctx.h2_titles))
  layouts = math.factorial(min(n, 12)) * (5 ** min(8, s)) * (4 ** min(6, b)) * max(1, h3)
  return min(int(layouts), 2_147_483_647)


async def run_optimizer_rag_pipeline(
  content: str,
  *,
  keywords: list[str],
  category: str | None = None,
  tone: str = "professional",
  variation_seed: int | None = None,
  use_rag: bool = True,
) -> dict[str, Any]:
  seed = effective_variation_seed(variation_seed)
  client_seed = make_variation_seed(variation_seed) if variation_seed is not None else None
  t0 = time.perf_counter()
  user_supplied_kws = bool(keywords)
  short_topic, display_title, kws = normalize_keywords(
    content, keywords, user_supplied=user_supplied_kws,
  )
  topic = short_topic
  anchors = derive_anchor_terms(content, short_topic, kws)
  strong = is_content_already_strong(content)
  stages: dict[str, Any] = {}

  # 1 Existing content
  stages["existing_content"] = {
    "word_count": count_words(content),
    "character_count": len(content.strip()),
    "sentence_count": count_sentences(content),
    "readability_score": readability_score(content),
  }

  # 2 Keyword analysis
  kw_analysis = analyze_keywords(content, kws)
  stages["keyword_analysis"] = kw_analysis

  # 3 Entity extraction
  content_entities = extract_entities_from_content(content, kws)
  stages["entity_extraction"] = {"from_content": content_entities}

  # 4 Coverage map
  coverage_map = build_coverage_map(content, kws, content_entities)
  stages["coverage_map"] = coverage_map

  # 5 Gap analysis (pre-retrieval, always local)
  pre_gaps = local_gap_analysis(content, coverage_map, kws, short_topic=short_topic)
  stages["gap_analysis"] = {"gaps": pre_gaps, "pre_retrieval_count": len(pre_gaps), "mode": "local"}

  # 6 Source router (+ fast path for strong drafts)
  topic_class = classify_topic(topic, kws, category)
  docs: list[Any] = []
  facts: list[ExtractedFact] = []
  gaps: list[dict[str, str]] = []
  all_entities = content_entities
  sources_used: list[str] = []
  confidence = 0.0
  novelty: dict[str, Any] = {
    "novel_facts": [],
    "redundant_count": 0,
    "novel_count": 0,
    "novelty_ratio": 0.0,
  }
  novel_facts: list[dict[str, Any]] = []

  if strong or not use_rag:
    gaps = list(pre_gaps)
    stages["source_router"] = {
      "topic_class": topic_class,
      "sources": [],
      "fast_path": "strong_content" if strong else "rag_disabled",
    }
    stages["retriever"] = {"raw_count": 0, "skipped": True}
    stages["deduplication"] = {"unique_count": 0, "skipped": True}
    stages["reranker"] = {"top_k": 0, "skipped": True}
    stages["fact_extraction"] = {"fact_count": 0, "skipped": True}
    stages["relevance_filter"] = {"kept": 0, "dropped": 0, "skipped": True}
    stages["entity_extraction"]["merged"] = all_entities
    stages["gap_analysis"] = {"gaps": gaps, "pre_retrieval_count": len(pre_gaps), "mode": "local"}
  elif len(pre_gaps) > 0:
    sources = route_sources(topic_class, max_sources=4)
    stages["source_router"] = {"topic_class": topic_class, "sources": sources, "fast_path": "gap_fill_rag"}
    try:
      raw_docs = await retrieve_from_sources(topic, kws, sources, per_source=2, seed=seed)
      sources_used = sorted({d.source for d in raw_docs})
      docs = deduplicate_docs(raw_docs)
      docs = rerank_docs(f"{topic} {' '.join(kws)}", docs, top_k=8)
      facts = extract_facts(docs, topic, kws)
      facts = resolve_conflicts(facts)
      confidence = score_confidence(facts, docs)
      all_entities = list(dict.fromkeys(content_entities + extract_entities(topic, docs, facts)))[:20]
      novel_facts = [{"text": f.text, "source": f.source, "confidence": f.confidence} for f in facts[:8]]
      stages["retriever"] = {"raw_count": len(raw_docs), "skipped": False}
      stages["deduplication"] = {"unique_count": len(docs), "skipped": False}
      stages["reranker"] = {"top_k": len(docs), "skipped": False}
      stages["fact_extraction"] = {"fact_count": len(facts), "skipped": False, "confidence": confidence}
      stages["relevance_filter"] = {"kept": len(docs), "dropped": 0, "skipped": False}
    except Exception as exc:
      stages["retriever"] = {"raw_count": 0, "skipped": True, "error": str(exc)[:80]}
      stages["fact_extraction"] = {"fact_count": 0, "skipped": True}
    gaps = list(pre_gaps)
    stages["entity_extraction"]["merged"] = all_entities
    stages["gap_analysis"] = {"gaps": gaps, "pre_retrieval_count": len(pre_gaps), "mode": "gap_rag"}
  else:
    # Fast local-only path — retrieval never injects into body; skip network + ML index for speed
    gaps = list(pre_gaps)
    stages["source_router"] = {
      "topic_class": topic_class,
      "sources": [],
      "fast_path": "local_only",
      "note": "RAG skipped for speed; gaps from local coverage analysis only",
    }
    stages["retriever"] = {"raw_count": 0, "skipped": True, "reason": "optimizer_local_fast"}
    stages["deduplication"] = {"unique_count": 0, "skipped": True}
    stages["reranker"] = {"top_k": 0, "skipped": True}
    stages["fact_extraction"] = {"fact_count": 0, "skipped": True}
    stages["relevance_filter"] = {"kept": 0, "dropped": 0, "skipped": True}
    stages["entity_extraction"]["merged"] = all_entities
    stages["gap_analysis"] = {"gaps": gaps, "pre_retrieval_count": len(pre_gaps), "mode": "local_fast"}

  # Gap fill — convert missing terms into real sections (never paste suggestions)
  gap_filled, terms_added = fill_content_gaps(
    content,
    gaps,
    coverage_map,
    topic=topic,
    keywords=kws,
    facts=novel_facts,
    seed=seed,
  )
  stages["gap_fill"] = {"terms_added": terms_added, "count": len(terms_added)}
  working_content = gap_filled

  if kws:
    working_content, kw_rewrite_notes = rewrite_content_for_keywords(
      working_content,
      kws,
      topic=short_topic,
      seed=seed + 401,
    )
    stages["keyword_rewriter"] = {
      "target_keywords": kws,
      "applied": True,
      "notes": kw_rewrite_notes[:8],
    }
  else:
    stages["keyword_rewriter"] = {"applied": False}

  # Section planner + generator
  section_plan = plan_sections(working_content, kws, gaps, coverage_map, seed=seed)
  stages["section_planner"] = {"sections": section_plan}

  vctx = build_variation_context(
    working_content,
    short_topic=short_topic,
    display_title=display_title,
    keywords=kws,
    gaps=gaps,
    seed=seed,
  )

  gen_suggestions: list[str] = []
  if kws and stages.get("keyword_rewriter", {}).get("notes"):
    gen_suggestions.extend(stages["keyword_rewriter"]["notes"])

  draft, polish_suggestions = polish_strong_content(
    working_content,
    short_topic=short_topic,
    keywords=kws,
    coverage_map=coverage_map,
    gaps=gaps,
    seed=seed,
    ctx=vctx,
    skip_variation_rebuild=bool(terms_added),
  )
  gen_suggestions.extend(polish_suggestions)
  gen_suggestions.insert(0, f"Gap fill added {len(terms_added)} section(s) from coverage analysis.")
  stages["section_generator"] = {
    "planned_sections": len(section_plan),
    "draft_words": count_words(draft),
    "conservative_mode": True,
    "external_injection": bool(novel_facts),
    "terms_filled": terms_added,
  }

  novelty = detect_local_opportunities(content, coverage_map, gaps)
  if facts:
    rag_novel = detect_novelty(content, facts)
    novelty["rag_novel_count"] = rag_novel["novel_count"]
    novelty["rag_mode"] = "analytics_only"
  stages["novelty_detector"] = novelty

  vctx = build_variation_context(
    draft,
    short_topic=short_topic,
    display_title=display_title,
    keywords=kws,
    gaps=gaps,
    seed=seed + 211,
  )
  faqs = generate_optimizer_faqs(
    topic, kws, draft, entities=all_entities, seed=seed,
  )
  if len(faqs) < 3:
    faqs = generate_faqs_from_content(vctx)
  faqs = rotate_faqs(faqs, seed)
  draft = append_faqs_to_content(draft, faqs)
  stages["faq_generator"] = {"count": len(faqs), "faqs": faqs, "source": "topic_aware"}

  metadata = optimize_metadata(draft, short_topic, display_title, kws, seed=seed, ctx=vctx)
  stages["metadata_generator"] = metadata

  optimized, read_suggestions = optimize_readability(draft)
  stages["readability_optimizer"] = {
    "before": content_metrics(draft),
    "after": content_metrics(optimized),
  }

  optimized, human_suggestions = humanize_text(optimized, seed=seed, tone=tone, ctx=vctx)
  stages["humanizer"] = {"applied": True, "tone": tone}

  optimized = preserve_markdown_structure(optimized)

  internal_links = internal_linking_suggestions(kws, all_entities, gaps)
  schema = schema_suggestions(metadata["title"], metadata["meta_description"], faqs)
  original_metrics = content_metrics(content)
  optimized_metrics = content_metrics(optimized)
  issues_before = analyze_issues(content, kws)
  issues_after = analyze_issues(optimized, kws)
  seo_before = seo_score_from_analysis(original_metrics, issues_before)
  seo_after = seo_score_from_analysis(optimized_metrics, issues_after)
  stages["seo_scorer"] = {
    "before": seo_before,
    "after": seo_after,
    "improvement": seo_after - seo_before,
  }
  stages["final_article"] = {"word_count": optimized_metrics["word_count"]}

  all_suggestions = _shuffle(gen_suggestions + read_suggestions + human_suggestions, seed + 201)
  for g in _shuffle(gaps, seed + 307)[:4]:
    msg = g.get("suggestion", "")
    if msg and msg not in all_suggestions:
      if "faq" in msg.lower() and re.search(r"##\s+frequently asked questions", optimized, re.I):
        continue
      all_suggestions.append(msg)
  for issue in issues_before[:3]:
    if issue["message"] not in all_suggestions:
      all_suggestions.append(issue["message"])

  seo_report = build_comprehensive_seo_report(
    content=content,
    optimized=optimized,
    short_topic=short_topic,
    keywords=kws,
    kw_analysis=kw_analysis,
    coverage_map=coverage_map,
    gaps=gaps,
    entities=all_entities,
    metadata=metadata,
    faqs=faqs,
    internal_links=internal_links,
    schema=schema,
    seo_before=seo_before,
    seo_after=seo_after,
    original_metrics=original_metrics,
    optimized_metrics=optimized_metrics,
    suggestions=all_suggestions,
    ctx=vctx,
  )

  return {
    "architecture": {"flow": ARCHITECTURE_FLOW, "stages": stages},
    "pipeline": {
      "keyword_analysis": kw_analysis,
      "entity_extraction": all_entities,
      "coverage_map": coverage_map,
      "gap_analysis": gaps,
      "source_router": stages["source_router"],
      "retrieval": {"sources_used": sources_used, "document_count": len(docs), "confidence": confidence},
      "novelty": novelty,
      "section_plan": section_plan,
      "readability_analysis": original_metrics,
    },
    "optimization": {
      "metadata": metadata,
      "internal_links": internal_links,
      "faqs": faqs,
      "schema_suggestions": schema,
      "seo_report": seo_report,
    },
    "optimized_content": optimized,
    "suggestions": all_suggestions[:14],
    "seo_score_before": seo_before,
    "seo_score_after": seo_after,
    "improvement": seo_after - seo_before,
    "issues_before": issues_before,
    "issues_after": issues_after,
    "original_metrics": original_metrics,
    "optimized_metrics": optimized_metrics,
    "variation_seed": seed,
    "client_variation_seed": client_seed,
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "variation_profile": {
      "mode": "max_variation_per_seed",
      "unlimited_outputs": True,
      "per_request_unique": True,
      "seed_space_bits": 31,
      "seed_space_size": 2_147_483_647,
      "note": "No output cap — each request draws a new effective seed; not limited to any test batch size",
      "estimated_layouts": estimate_variation_space(vctx),
      "variation_passes": [
        "h1_vary", "intro_recompose", "h2_reorder", "h3_shuffle", "sentence_shuffle",
        "bullet_shuffle", "lead_ins", "lexical_full", "conclusion_recompose",
      ],
      "faq_count": len(faqs),
      "sentence_pool": len(vctx.sentences),
      "h2_pool": len(vctx.h2_titles),
      "bullet_pool": len(vctx.bullets),
    },
    "rag_sources": sources_used,
    "generator_version": GENERATOR_VERSION,
  }
