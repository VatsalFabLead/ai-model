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
from app.engine.seo_retrieval_engine import (
  StrictFact,
  anchor_terms,
  assign_facts_to_sections,
  build_dynamic_outline,
  chunk_documents,
  cross_encoder_rerank_chunks,
  detect_nsfw_topic,
  disambiguate_entities_embedding,
  entity_candidates,
  extract_facts_strict,
  filter_relevant_documents,
  generate_paa_faqs,
  generate_safe_metadata,
  polluted_fact,
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
  """Strict fact extraction — word-boundary match, chunk rerank, no pollution fallback."""
  anchors = anchor_terms(topic, keywords)
  nsfw = detect_nsfw_topic(topic, keywords)
  if nsfw.get("skip_open_retrieval"):
    return []

  filtered_docs = filter_relevant_documents(docs, topic, keywords)
  chunks = chunk_documents(filtered_docs)
  query = f"{topic} {' '.join(keywords)}".strip()
  ranked = cross_encoder_rerank_chunks(query, chunks, anchors)
  strict = extract_facts_strict(ranked, topic, keywords)

  facts: list[ExtractedFact] = []
  for sf in strict:
    if polluted_fact(sf.text, anchors, topic=topic, keywords=keywords):
      continue
    facts.append(ExtractedFact(
      text=sf.text,
      source=sf.source,
      confidence=sf.confidence,
      entities=sf.entities,
    ))
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
  intent: str = "informational",
  content_profile: str = "general",
) -> dict[str, Any]:
  """Build article from verified facts — one unique fact per section."""
  seed = rag.variation_seed
  domain = detect_domain(topic, keywords)
  kw = expand_keywords(topic, keywords, domain)
  primary = kw["primary"]

  outline = build_dynamic_outline(
    topic, primary, profile=content_profile, intent=intent, seed=seed,
  )
  if not outline:
    outline = build_structured_outline(topic, primary, domain=domain, category=category, seed=seed)

  facts = rag.facts
  strict_facts = [
    StrictFact(text=f.text, source=f.source, confidence=f.confidence, chunk_score=f.confidence)
    for f in facts
  ]

  title = outline[0]["text"] if outline else f"{topic}: Complete Guide"
  h2_headings = [
    o["text"] for o in outline
    if o.get("level") == "h2" and o["text"].lower() not in ("introduction", "conclusion")
  ]
  section_facts = assign_facts_to_sections(h2_headings, strict_facts, topic=topic, primary=primary)

  meta = generate_safe_metadata(title, "", primary, topic)

  intro_pool = [
    f"This guide explains **{primary}** in the context of **{topic}** with structured, "
    "search-aligned information readers can trust.",
    f"**{topic}** covers important considerations around **{primary}**. "
    "Below is a clear overview organized by section.",
  ]
  sections: list[str] = [f"# {title}", ""]
  if strict_facts:
    sections.append(f"> **Quick answer:** {strict_facts[0].text[:260]}\n")
  sections.extend(["## Introduction", "", _pick(intro_pool, seed), ""])

  for h2 in h2_headings:
    sections.append(f"## {h2}")
    sections.append("")
    body = section_facts.get(h2, "")
    sections.append(body)
    sections.append("")

  sections.extend([
    "## Conclusion",
    "",
    f"Use this guide to make informed decisions about **{primary}** related to **{topic}**. "
    "Verify details with reputable sources and prioritize safety and clarity.",
  ])

  article = re.sub(r"\n{3,}", "\n\n", "\n".join(sections)).strip()
  meta = generate_safe_metadata(title, article, primary, topic)

  word_count = len(re.findall(r"\b[\w'-]+\b", article))
  faqs: list[dict[str, str]] = []

  if word_count < max(200, target_words // 2) or content_profile in ("adult_services", "local_services", "ambiguous_escort"):
    if content_profile in ("adult_services", "local_services", "ambiguous_escort"):
      h2_outline = [o for o in outline if o.get("level") == "h2"]
      article = build_profile_article_from_outline(
        topic, primary, outline,
        profile=content_profile if content_profile != "ambiguous_escort" else "adult_services",
        intent=intent,
      )
      title = outline[0]["text"] if outline else title
      faqs = generate_paa_faqs(
        topic, primary, strict_facts, content_profile, intent,
        keywords=keywords, domain=domain, seed=seed,
      )
    else:
      from app.engine.seo_content_domains import build_rich_content

      rich = build_rich_content(
        topic, keywords, category=category, tone=tone, audience=audience, seed=seed,
      )
      title = rich["metadata"]["title"]
      outline = rich.get("outline", outline)
      article = rich["content"]["article"]
      meta = generate_safe_metadata(title, article, primary, topic)
      faqs = list(rich.get("faqs", []))

  if not faqs:
    faqs = generate_paa_faqs(
      topic, primary, strict_facts, content_profile, intent,
      keywords=keywords, domain=domain, seed=seed,
    )

  return {
    "metadata": {"title": title[:70], "meta_description": meta},
    "keywords": kw,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
    "domain": domain,
    "variation_seed": seed,
    "content_profile": content_profile,
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
  nsfw = detect_nsfw_topic(topic, keywords)
  sources = route_sources(topic_class, max_sources=max_sources)

  raw_docs: list[OpenDoc] = []
  if not nsfw.get("skip_open_retrieval"):
    raw_docs = await retrieve_from_sources(
      topic, keywords, sources, per_source=2, seed=seed,
    )
    raw_docs = filter_relevant_documents(raw_docs, topic, keywords)

  sources_hit = sorted({d.source for d in raw_docs})
  docs = deduplicate_docs(raw_docs)

  anchors = anchor_terms(topic, keywords)
  chunks = chunk_documents(docs)
  query = f"{topic} {' '.join(keywords)}".strip()
  ranked_chunks = cross_encoder_rerank_chunks(query, chunks, anchors)
  if ranked_chunks:
    docs = [
      OpenDoc(
        doc_id=f"{ch.source}:{i}",
        source=ch.source,
        title=ch.title,
        text=ch.text,
        score=ch.score,
      )
      for i, ch in enumerate(ranked_chunks[:top_k])
    ]
  docs = rerank_docs(query, docs, top_k=top_k)

  facts = extract_facts(docs, topic, keywords)
  facts = resolve_conflicts(facts)

  cands = entity_candidates(topic, keywords)
  disambig = disambiguate_entities_embedding(topic, keywords, cands)
  entities = extract_entities(topic, docs, facts)
  if disambig.get("selected"):
    entities = [disambig["selected"]] + [e for e in entities if e != disambig["selected"]]

  confidence = score_confidence(facts, docs)
  evidence = _evidence_block(docs, facts)

  return RagPipelineResult(
    topic_class=topic_class,
    sources_routed=sources,
    sources_used=sources_hit,
    documents=docs,
    facts=facts,
    entities=entities[:15],
    confidence=confidence,
    evidence_context=evidence,
    variation_seed=seed,
  )
