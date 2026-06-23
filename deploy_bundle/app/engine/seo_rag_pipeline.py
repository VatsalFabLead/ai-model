"""SEO Content RAG pipeline — production architecture on free open datasets.

User Query → Topic Classifier → Source Router → Retrieve Top-k
→ Deduplication → Embedding Reranker → Fact Extractor → Conflict Resolver
→ Confidence Scoring → Entity Extraction → Synthesis (custom LLM optional)

No GPT/Claude/Gemini APIs.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.engine.open_data_retrieval import (
  OpenDoc,
  SOURCE_ROUTES,
  retrieve_from_sources,
)
from app.engine.seo_content_domains import (
  build_structured_outline,
  detect_domain,
  expand_keywords,
  make_variation_seed,
)

_SOURCE_PRIORITY = {
  "wikipedia": 0.95,
  "wikidata": 0.88,
  "dbpedia": 0.86,
  "semantic_scholar": 0.84,
  "arxiv": 0.82,
  "stackexchange": 0.8,
  "gooaq": 0.78,
  "squad": 0.76,
  "dolly": 0.75,
  "conceptnet": 0.72,
  "gdelt": 0.7,
  "fineweb": 0.68,
  "c4": 0.66,
  "local_faiss": 0.65,
}

_TOPIC_SIGNALS: dict[str, list[str]] = {
  "technical": [
    "api", "code", "programming", "software", "algorithm", "machine learning",
    "python", "javascript", "flutter", "database", "cloud", "ai model",
  ],
  "news": [
    "news", "latest", "today", "breaking", "update", "2025", "2026", "trend",
    "announcement", "report",
  ],
  "health_fitness": [
    "workout", "exercise", "fitness", "health", "diet", "nutrition", "yoga",
    "muscle", "cardio", "wellness", "beginner workout",
  ],
  "programming": [
    "python", "javascript", "code", "bug", "function", "class", "api",
    "stackoverflow", "developer", "programming",
  ],
  "how_to": [
    "how to", "step by step", "guide", "tutorial", "learn", "beginner",
  ],
  "business": [
    "marketing", "seo", "business", "sales", "startup", "revenue", "brand",
  ],
  "enterprise": [
    "erp", "crm", "enterprise resource", "inventory management", "manufacturing erp",
    "supply chain", "accounting software", "procurement", "warehouse management",
  ],
}


@dataclass
class ExtractedFact:
  text: str
  source: str
  confidence: float
  entities: list[str] = field(default_factory=list)


@dataclass
class RagPipelineResult:
  topic_class: str
  sources_routed: list[str]
  sources_used: list[str]
  documents: list[OpenDoc]
  facts: list[ExtractedFact]
  entities: list[str]
  confidence: float
  evidence_context: str
  variation_seed: int


def classify_topic(topic: str, keywords: list[str], category: str | None = None) -> str:
  text = f"{topic} {' '.join(keywords)} {category or ''}".lower()
  if any(s in text for s in _TOPIC_SIGNALS.get("enterprise", [])):
    return "enterprise"
  if any(s in text for s in _TOPIC_SIGNALS["programming"]):
    return "programming"
  if any(s in text for s in _TOPIC_SIGNALS["health_fitness"]):
    return "health_fitness"
  if any(s in text for s in _TOPIC_SIGNALS["technical"]):
    return "technical"
  if any(s in text for s in _TOPIC_SIGNALS["news"]):
    return "news"
  if any(s in text for s in _TOPIC_SIGNALS["how_to"]) or (category or "") == "how_to_guide":
    return "how_to"
  if any(s in text for s in _TOPIC_SIGNALS["business"]):
    return "business"
  return "general"


def route_sources(topic_class: str, *, max_sources: int = 6) -> list[str]:
  route = SOURCE_ROUTES.get(topic_class, SOURCE_ROUTES["general"])
  return route[:max_sources]


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


def deduplicate_docs(docs: list[OpenDoc], *, threshold: float = 0.72) -> list[OpenDoc]:
  kept: list[OpenDoc] = []
  for doc in sorted(docs, key=lambda d: d.score, reverse=True):
    if any(_shingle_jaccard(doc.text, k.text) >= threshold for k in kept):
      continue
    kept.append(doc)
  return kept


def _embed_query_and_docs(query: str, docs: list[OpenDoc]) -> list[OpenDoc]:
  if not docs:
    return []
  try:
    from app.engine.plagiarism_engine import embed_texts

    texts = [query] + [f"{d.title}. {d.text}" for d in docs]
    vecs = embed_texts(texts)
    if vecs is None or len(vecs) < 2:
      return docs
    q_vec = vecs[0]
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
      q_vec = q_vec / q_norm
    reranked: list[OpenDoc] = []
    for i, doc in enumerate(docs):
      d_vec = vecs[i + 1]
      d_norm = np.linalg.norm(d_vec)
      sim = float(np.dot(q_vec, d_vec / d_norm)) if d_norm > 0 else 0.0
      src_boost = _SOURCE_PRIORITY.get(doc.source, 0.6)
      doc.score = 0.55 * sim + 0.25 * doc.score + 0.2 * src_boost
      reranked.append(doc)
    reranked.sort(key=lambda d: d.score, reverse=True)
    return reranked
  except Exception:
    for doc in docs:
      doc.score = doc.score * 0.5 + _SOURCE_PRIORITY.get(doc.source, 0.5) * 0.5
    return sorted(docs, key=lambda d: d.score, reverse=True)


def rerank_docs(
  query: str,
  docs: list[OpenDoc],
  *,
  top_k: int = 8,
  use_embeddings: bool = True,
) -> list[OpenDoc]:
  if use_embeddings and len(docs) > 2:
    ranked = _embed_query_and_docs(query, docs)
  else:
    for doc in docs:
      doc.score = doc.score * 0.5 + _SOURCE_PRIORITY.get(doc.source, 0.5) * 0.5
    ranked = sorted(docs, key=lambda d: d.score, reverse=True)
  return ranked[:top_k]


def _split_sentences(text: str) -> list[str]:
  parts = re.split(r"(?<=[.!?])\s+", text)
  return [p.strip() for p in parts if len(p.strip()) > 25]


def extract_facts(docs: list[OpenDoc], topic: str, keywords: list[str]) -> list[ExtractedFact]:
  terms = {t.lower() for t in re.findall(r"\w+", f"{topic} {' '.join(keywords)}") if len(t) > 2}
  facts: list[ExtractedFact] = []
  seen: set[str] = set()
  for doc in docs:
    for sent in _split_sentences(doc.text):
      low = sent.lower()
      if not terms or not any(t in low for t in terms):
        if len(terms) > 2 and sum(1 for t in terms if t in low) < 1:
          continue
      key = low[:80]
      if key in seen:
        continue
      seen.add(key)
      conf = doc.score * _SOURCE_PRIORITY.get(doc.source, 0.6)
      ents = [w for w in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", sent)][:5]
      facts.append(
        ExtractedFact(text=sent, source=doc.source, confidence=conf, entities=ents)
      )
  facts.sort(key=lambda f: f.confidence, reverse=True)
  if not facts and docs:
    for doc in docs[:4]:
      for sent in _split_sentences(doc.text)[:4]:
        facts.append(
          ExtractedFact(text=sent, source=doc.source, confidence=doc.score * 0.7)
        )
    facts = resolve_conflicts(facts)
  return facts[:24]


def resolve_conflicts(facts: list[ExtractedFact]) -> list[ExtractedFact]:
  """Drop near-duplicate facts; prefer higher-confidence source."""
  kept: list[ExtractedFact] = []
  for fact in facts:
    if any(_shingle_jaccard(fact.text, k.text) > 0.82 for k in kept):
      continue
    kept.append(fact)
  return kept


def score_confidence(facts: list[ExtractedFact], docs: list[OpenDoc]) -> float:
  if not facts:
    return 0.25
  avg_fact = sum(f.confidence for f in facts[:8]) / min(8, len(facts))
  src_diversity = len({d.source for d in docs}) / max(1, len(docs))
  return round(min(0.98, 0.4 * avg_fact + 0.35 * src_diversity + 0.25 * min(1.0, len(docs) / 6)), 3)


def extract_entities(topic: str, docs: list[OpenDoc], facts: list[ExtractedFact]) -> list[str]:
  entities: list[str] = []
  seen: set[str] = set()
  for src in [topic] + [d.title for d in docs] + [e for f in facts for e in f.entities]:
    for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,3}\b", src):
      k = phrase.lower()
      if k not in seen and len(phrase) > 2:
        seen.add(k)
        entities.append(phrase)
  return entities[:15]


def _evidence_block(docs: list[OpenDoc], facts: list[ExtractedFact]) -> str:
  lines = ["## Retrieved evidence (open datasets)"]
  for doc in docs[:6]:
    lines.append(f"- [{doc.source}] {doc.title}: {_clip(doc.text, 280)}")
  lines.append("\n## Key facts")
  for i, f in enumerate(facts[:12], 1):
    lines.append(f"{i}. ({f.source}) {f.text}")
  return "\n".join(lines)


def _clip(text: str, n: int) -> str:
  text = re.sub(r"\s+", " ", text.strip())
  return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _pick(items: list[str], seed: int) -> str:
  return items[seed % len(items)] if items else ""


def synthesize_structured_content(
  topic: str,
  keywords: list[str],
  rag: RagPipelineResult,
  *,
  category: str,
  tone: str,
  audience: str | None,
  target_words: int,
) -> dict[str, Any]:
  """Build unique article from RAG evidence — not static templates."""
  seed = rag.variation_seed
  domain = detect_domain(topic, keywords)
  kw = expand_keywords(topic, keywords, domain)
  primary = kw["primary"]
  outline = build_structured_outline(topic, primary, domain=domain, category=category, seed=seed)

  facts = rag.facts
  fact_texts = [f.text for f in facts]

  # Title variants from evidence
  title = _pick([
    f"{topic}: A Complete Guide",
    f"{topic} — Expert Guide for Beginners",
    f"{primary.title()}: Everything You Need to Know",
    f"The Ultimate Guide to {primary.title()}",
  ], seed)
  if rag.entities:
    title = _pick([
      f"{topic}: A Professional Guide to {rag.entities[0]}",
      title,
    ], seed + 1)

  meta_bits = fact_texts[:2] or [f"Learn about {primary} with practical, evidence-based guidance."]
  meta = _clip(
    f"Discover {primary} with actionable tips and expert insights. "
    + " ".join(meta_bits)[:120],
    160,
  )

  # Build article sections from facts
  intro_pool = [
    f"Understanding **{topic}** helps readers make informed decisions. "
    "This guide combines verified open-data sources with practical advice.",
    f"**{topic}** is a subject many beginners want to master. "
    "Below is a structured, evidence-informed overview you can apply right away.",
    f"Whether you are new to **{primary}** or refining your approach, "
    "this article distills reliable information into clear, actionable steps.",
  ]
  sections: list[str] = [f"# {outline[0]['text'] if outline else title}", "", "## Introduction", "", _pick(intro_pool, seed + 2), ""]

  h2_outline = [o for o in outline if o.get("level") == "h2" and o["text"].lower() != "introduction"]
  fact_idx = 0
  for i, h2 in enumerate(h2_outline):
    if h2["text"].lower() == "conclusion":
      continue
    sections.append(f"## {h2['text']}")
    sections.append("")
    chunk: list[str] = []
    while fact_idx < len(fact_texts) and len(" ".join(chunk).split()) < max(60, target_words // max(1, len(h2_outline))):
      chunk.append(fact_texts[fact_idx])
      fact_idx += 1
    if chunk:
      sections.append(" ".join(chunk[:3]))
    else:
      sections.append(
        _pick([
          f"When exploring {primary}, focus on fundamentals, consistent practice, and measurable progress.",
          f"Research on {primary} highlights practical steps, common pitfalls, and long-term benefits.",
        ], seed + i)
      )
    sections.append("")

  # H3 subsections from outline
  h3_items = [o for o in outline if o.get("level") == "h3"]
  if h3_items and fact_idx < len(fact_texts):
    parent_h2 = next((o for o in h2_outline if "plan" in o["text"].lower() or "step" in o["text"].lower()), None)
    if parent_h2:
      sections.append(f"## {parent_h2['text']}")
      sections.append("")
    for j, h3 in enumerate(h3_items[:5]):
      sections.append(f"### {h3['text']}")
      sections.append("")
      if fact_idx < len(fact_texts):
        sections.append(fact_texts[fact_idx])
        fact_idx += 1
      else:
        sections.append(f"Apply proven techniques for {h3['text'].lower()} as part of your {primary} routine.")
      sections.append("")

  sections.extend([
    "## Conclusion",
    "",
    _pick([
      f"**{primary.title()}** becomes achievable with the right structure and consistent effort. "
      "Use the steps above, track your progress, and refine your approach over time.",
      f"By applying these evidence-based principles to **{topic}**, you can build lasting results. "
      "Start small, stay consistent, and improve week by week.",
    ], seed + 7),
  ])

  article = "\n".join(sections).strip()
  article = re.sub(r"\n{3,}", "\n\n", article)

  domain = detect_domain(topic, keywords)
  word_count = len(re.findall(r"\b[\w'-]+\b", article))
  faqs: list[dict[str, str]] = []

  if word_count < max(200, target_words // 2) or domain in ("enterprise", "fitness"):
    from app.engine.seo_content_domains import build_rich_content

    rich = build_rich_content(
      topic, keywords, category=category, tone=tone, audience=audience, seed=seed,
    )
    title = rich["metadata"]["title"]
    meta = rich["metadata"]["meta_description"]
    outline = rich.get("outline", outline)
    rich_article = rich["content"]["article"]
    if fact_texts:
      if "## Introduction" in rich_article:
        head, tail = rich_article.split("## Introduction", 1)
        article = f"{head.rstrip()}\n\n## Introduction\n\n{fact_texts[0]}\n\n{tail.lstrip()}"
      else:
        article = f"## Introduction\n\n{fact_texts[0]}\n\n{rich_article}"
    else:
      article = rich_article
    faqs = list(rich.get("faqs", []))

  if not faqs:
    faq_questions = [
      f"What is {primary}?",
      f"How do I get started with {primary}?",
      f"What are the benefits of {primary}?",
      f"How long does it take to see results with {primary}?",
      f"Who should focus on {primary}?",
    ]
    for i, q in enumerate(faq_questions[:5]):
      if i < len(fact_texts):
        faqs.append({"question": q, "answer": _clip(fact_texts[i], 300)})
      else:
        faqs.append({
          "question": q,
          "answer": (
            f"{primary.title()} offers practical value for anyone learning about {topic}. "
            "Start with fundamentals and build gradually."
          ),
        })

  return {
    "metadata": {"title": title[:70], "meta_description": meta},
    "keywords": kw,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
    "domain": domain,
    "variation_seed": seed,
    "rag": {
      "topic_class": rag.topic_class,
      "confidence": rag.confidence,
      "sources_used": rag.sources_used,
      "document_count": len(rag.documents),
      "fact_count": len(rag.facts),
      "entities": rag.entities[:10],
    },
  }


async def run_seo_rag_pipeline(
  topic: str,
  keywords: list[str],
  *,
  category: str | None = None,
  variation_seed: int | None = None,
  top_k: int = 8,
  max_sources: int = 4,
) -> RagPipelineResult:
  seed = make_variation_seed(variation_seed)
  topic_class = classify_topic(topic, keywords, category)
  sources = route_sources(topic_class, max_sources=max_sources)

  raw_docs = await retrieve_from_sources(
    topic, keywords, sources, per_source=2, seed=seed,
  )
  sources_hit = sorted({d.source for d in raw_docs})
  docs = deduplicate_docs(raw_docs)
  docs = rerank_docs(topic, docs, top_k=top_k)

  # Rotate top docs by seed for variety between runs
  if len(docs) > 2:
    offset = seed % len(docs)
    docs = docs[offset:] + docs[:offset]

  facts = extract_facts(docs, topic, keywords)
  facts = resolve_conflicts(facts)
  confidence = score_confidence(facts, docs)
  entities = extract_entities(topic, docs, facts)
  evidence = _evidence_block(docs, facts)

  return RagPipelineResult(
    topic_class=topic_class,
    sources_routed=sources,
    sources_used=sources_hit,
    documents=docs,
    facts=facts,
    entities=entities,
    confidence=confidence,
    evidence_context=evidence,
    variation_seed=seed,
  )
