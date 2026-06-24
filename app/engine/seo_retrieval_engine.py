"""SEO retrieval quality — chunking, cross-encoder rerank, strict facts, disambiguation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.engine.open_data_retrieval import OpenDoc

_MIN_CHUNK_CHARS = 80
_MAX_CHUNK_CHARS = 520
_RELEVANCE_THRESHOLD = 0.38
_FACT_RELEVANCE_THRESHOLD = 0.42

_AMBIGUOUS_TERMS = frozenset({
  "escort", "apple", "java", "mercury", "jaguar", "amazon", "delta", "target",
  "shell", "oracle", "accent", "matrix", "fusion", "eclipse",
})

_NSFW_SERVICE_SIGNALS = (
  "escort service", "escort services", "companion service", "companion services",
  "call girl", "adult entertainment", "escort agency", "escort agencies",
  "bangalore escort", "mumbai escort", "delhi escort", "dating escort",
)

_AUTOMOTIVE_SIGNALS = ("ford escort", "car escort", "vehicle escort", "motor escort")

_SECURITY_SIGNALS = (
  "security escort", "vip escort", "police escort", "military escort",
  "armed escort", "convoy escort",
)

_ENTITY_CANDIDATES: dict[str, list[str]] = {
  "escort": [
    "Escort services",
    "Ford Escort",
    "Security escort",
    "Military escort",
    "VIP escort",
  ],
}

_STOPWORDS = frozenset({
  "the", "and", "for", "with", "from", "that", "this", "your", "about", "into",
  "guide", "best", "services", "service",
})


@dataclass
class TextChunk:
  text: str
  source: str
  title: str
  score: float = 0.0
  doc_score: float = 0.0


@dataclass
class StrictFact:
  text: str
  source: str
  confidence: float
  chunk_score: float
  entities: list[str] = field(default_factory=list)


def anchor_terms(topic: str, keywords: list[str]) -> list[str]:
  terms: list[str] = []
  seen: set[str] = set()
  for raw in [topic, *keywords]:
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", raw or ""):
      low = token.lower()
      if low in _STOPWORDS or low in seen:
        continue
      seen.add(low)
      terms.append(low)
  return terms[:12]


def term_in_text(term: str, text: str) -> bool:
  if not term or not text:
    return False
  return bool(re.search(rf"\b{re.escape(term)}\b", text, re.I))


def anchor_match_count(text: str, anchors: list[str]) -> int:
  return sum(1 for a in anchors if term_in_text(a, text))


def detect_content_profile(topic: str, keywords: list[str]) -> str:
  text = f"{topic} {' '.join(keywords)}".lower()
  if any(s in text for s in _NSFW_SERVICE_SIGNALS):
    return "adult_services"
  if any(s in text for s in _AUTOMOTIVE_SIGNALS):
    return "automotive"
  if any(s in text for s in _SECURITY_SIGNALS):
    return "security_escort"
  if re.search(r"\bescort\b", text):
    if any(w in text for w in ("agency", "companion", "bangalore", "mumbai", "delhi", "call", "booking")):
      return "adult_services"
    if any(w in text for w in ("ford", "car", "vehicle", "motor", "automobile")):
      return "automotive"
    if any(w in text for w in ("security", "vip", "military", "police", "convoy")):
      return "security_escort"
    return "ambiguous_escort"
  if any(w in text for w in ("bangalore", "mumbai", "delhi", "city", "local")):
    if "service" in text or "services" in text:
      return "local_services"
  return "general"


def detect_nsfw_topic(topic: str, keywords: list[str]) -> dict[str, Any]:
  profile = detect_content_profile(topic, keywords)
  is_adult = profile == "adult_services"
  is_ambiguous = profile == "ambiguous_escort" or (
    any(term_in_text(t, f"{topic} {' '.join(keywords)}") for t in _AMBIGUOUS_TERMS)
    and profile == "general"
  )
  return {
    "profile": profile,
    "is_adult": is_adult,
    "is_ambiguous": is_ambiguous,
    "skip_open_retrieval": is_adult or profile == "ambiguous_escort",
    "use_template_only": is_adult,
  }


def entity_candidates(topic: str, keywords: list[str]) -> list[str]:
  text = f"{topic} {' '.join(keywords)}".lower()
  cands: list[str] = []
  for key, options in _ENTITY_CANDIDATES.items():
    if term_in_text(key, text):
      cands.extend(options)
  if not cands and "escort" in text:
    cands = list(_ENTITY_CANDIDATES["escort"])
  return list(dict.fromkeys(cands))


def disambiguate_entities_embedding(
  topic: str,
  keywords: list[str],
  candidates: list[str],
) -> dict[str, Any]:
  if not candidates:
    return {"selected": topic, "confidence": 0.5, "candidates": []}
  query = f"{topic} {' '.join(keywords)}".strip()
  profile = detect_content_profile(topic, keywords)
  profile_map = {
    "adult_services": "Escort services",
    "automotive": "Ford Escort",
    "security_escort": "Security escort",
    "ambiguous_escort": "Escort services",
  }
  if profile in profile_map:
    selected = profile_map[profile]
    return {
      "selected": selected,
      "confidence": 0.82,
      "candidates": candidates,
      "method": "profile_classifier",
    }
  try:
    from app.engine.plagiarism_engine import generate_embeddings

    texts = [query] + candidates
    vecs = generate_embeddings(texts)
    if vecs is None or len(vecs) < 2:
      return {"selected": candidates[0], "confidence": 0.5, "candidates": candidates}
    q = vecs[0]
    qn = np.linalg.norm(q)
    if qn > 0:
      q = q / qn
    best_i, best_score = 0, -1.0
    for i, cvec in enumerate(vecs[1:]):
      cn = np.linalg.norm(cvec)
      sim = float(np.dot(q, cvec / cn)) if cn > 0 else 0.0
      if sim > best_score:
        best_score = sim
        best_i = i
    return {
      "selected": candidates[best_i],
      "confidence": round(best_score, 3),
      "candidates": candidates,
      "method": "embedding",
    }
  except Exception:
    return {"selected": candidates[0], "confidence": 0.5, "candidates": candidates}


def split_doc_chunks(doc: OpenDoc) -> list[TextChunk]:
  chunks: list[TextChunk] = []
  paragraphs = re.split(r"\n\s*\n", doc.text or "")
  if len(paragraphs) <= 1:
    paragraphs = re.split(r"(?<=[.!?])\s+", doc.text or "")
  buffer = ""
  for para in paragraphs:
    para = re.sub(r"\s+", " ", para.strip())
    if not para:
      continue
    if len(para) < _MIN_CHUNK_CHARS and buffer:
      buffer = f"{buffer} {para}".strip()
      continue
    if buffer:
      chunks.append(TextChunk(buffer, doc.source, doc.title, doc_score=doc.score))
      buffer = ""
    if len(para) <= _MAX_CHUNK_CHARS:
      chunks.append(TextChunk(para, doc.source, doc.title, doc_score=doc.score))
    else:
      sents = re.split(r"(?<=[.!?])\s+", para)
      buf2 = ""
      for s in sents:
        if len(buf2) + len(s) < _MAX_CHUNK_CHARS:
          buf2 = f"{buf2} {s}".strip()
        else:
          if buf2:
            chunks.append(TextChunk(buf2, doc.source, doc.title, doc_score=doc.score))
          buf2 = s
      if buf2:
        chunks.append(TextChunk(buf2, doc.source, doc.title, doc_score=doc.score))
  return chunks


def chunk_documents(docs: list[OpenDoc]) -> list[TextChunk]:
  out: list[TextChunk] = []
  for doc in docs:
    out.extend(split_doc_chunks(doc))
  return out


def _embed_similarity(query: str, texts: list[str]) -> list[float]:
  if not texts:
    return []
  try:
    from app.engine.plagiarism_engine import generate_embeddings

    vecs = generate_embeddings([query] + texts)
    if vecs is None or len(vecs) < 2:
      return [0.0] * len(texts)
    q = vecs[0]
    qn = np.linalg.norm(q)
    if qn > 0:
      q = q / qn
    scores: list[float] = []
    for vec in vecs[1:]:
      vn = np.linalg.norm(vec)
      scores.append(float(np.dot(q, vec / vn)) if vn > 0 else 0.0)
    return scores
  except Exception:
    return [0.0] * len(texts)


def cross_encoder_rerank_chunks(
  query: str,
  chunks: list[TextChunk],
  anchors: list[str],
  *,
  min_score: float = _RELEVANCE_THRESHOLD,
) -> list[TextChunk]:
  if not chunks:
    return []
  texts = [f"{c.title}. {c.text}" for c in chunks]
  sims = _embed_similarity(query, texts)
  ranked: list[TextChunk] = []
  for chunk, sim in zip(chunks, sims):
    anchor_boost = 0.12 * anchor_match_count(f"{chunk.title} {chunk.text}", anchors)
    title_boost = 0.1 if anchor_match_count(chunk.title, anchors) >= 1 else 0.0
    chunk.score = 0.72 * sim + 0.18 * chunk.doc_score + anchor_boost + title_boost
    if anchor_match_count(chunk.text, anchors) >= 1 or chunk.score >= min_score:
      ranked.append(chunk)
  ranked.sort(key=lambda c: c.score, reverse=True)
  return ranked


def filter_relevant_documents(
  docs: list[OpenDoc],
  topic: str,
  keywords: list[str],
) -> list[OpenDoc]:
  anchors = anchor_terms(topic, keywords)
  if not anchors:
    return docs
  kept: list[OpenDoc] = []
  for doc in docs:
    blob = f"{doc.title} {doc.text[:800]}"
    title_hits = anchor_match_count(doc.title, anchors)
    body_hits = anchor_match_count(blob, anchors)
    if title_hits >= 1 or body_hits >= max(1, len(anchors) // 2):
      kept.append(doc)
  return kept


def extract_facts_strict(
  chunks: list[TextChunk],
  topic: str,
  keywords: list[str],
  *,
  max_facts: int = 16,
) -> list[StrictFact]:
  anchors = anchor_terms(topic, keywords)
  facts: list[StrictFact] = []
  seen: set[str] = set()

  for chunk in chunks:
    if chunk.score < _FACT_RELEVANCE_THRESHOLD and anchor_match_count(chunk.text, anchors) < 1:
      continue
    for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
      sent = sent.strip()
      if len(sent) < 35 or len(sent) > 420:
        continue
      if anchor_match_count(sent, anchors) < 1:
        continue
      key = sent.lower()[:100]
      if key in seen:
        continue
      seen.add(key)
      ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", sent)[:4]
      facts.append(StrictFact(
        text=sent,
        source=chunk.source,
        confidence=round(min(0.98, chunk.score * 0.85 + 0.1), 3),
        chunk_score=chunk.score,
        entities=ents,
      ))
      if len(facts) >= max_facts:
        break
    if len(facts) >= max_facts:
      break

  facts.sort(key=lambda f: f.confidence, reverse=True)
  return facts


def assign_facts_to_sections(
  section_headings: list[str],
  facts: list[StrictFact],
  *,
  topic: str,
  primary: str,
) -> dict[str, str]:
  """One unique fact per section; never repeat the same sentence."""
  assignments: dict[str, str] = {}
  used: set[str] = set()
  fact_idx = 0
  for heading in section_headings:
    low = heading.lower()
    if low in ("introduction", "conclusion"):
      continue
    chosen = ""
    for _ in range(len(facts) or 1):
      if not facts:
        break
      candidate = facts[fact_idx % len(facts)]
      fact_idx += 1
      if candidate.text not in used:
        chosen = candidate.text
        used.add(candidate.text)
        break
    if not chosen:
      chosen = (
        f"When exploring {heading.lower()}, focus on verified information about "
        f"{primary} and avoid unrelated sources."
      )
    assignments[heading] = chosen
  return assignments


def build_dynamic_outline(
  topic: str,
  primary: str,
  *,
  profile: str,
  intent: str,
  seed: int,
) -> list[dict[str, str]]:
  def _h1() -> str:
    options = [
      f"{topic}: Complete Guide",
      f"{topic} — Professional Guide",
      f"{primary.title()}: Expert Overview",
    ]
    return options[seed % len(options)]

  if profile == "adult_services":
    return [
      {"level": "h1", "text": f"{topic} Guide"},
      {"level": "h2", "text": "Understanding Escort Services"},
      {"level": "h2", "text": "Safety and Privacy Considerations"},
      {"level": "h2", "text": "Types of Companion Services"},
      {"level": "h2", "text": "Choosing Reputable Agencies"},
      {"level": "h2", "text": "Booking Process"},
      {"level": "h2", "text": "Pricing Factors"},
      {"level": "h2", "text": "Frequently Asked Questions"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if profile == "local_services":
    return [
      {"level": "h1", "text": f"{topic} Guide"},
      {"level": "h2", "text": f"Overview of {primary.title()} in Your Area"},
      {"level": "h2", "text": "How to Evaluate Providers"},
      {"level": "h2", "text": "Safety and Discretion"},
      {"level": "h2", "text": "Booking and Communication"},
      {"level": "h2", "text": "Pricing and Expectations"},
      {"level": "h2", "text": "Frequently Asked Questions"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if profile == "automotive":
    return [
      {"level": "h1", "text": _h1()},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": "History and Model Overview"},
      {"level": "h2", "text": "Specifications and Features"},
      {"level": "h2", "text": "Maintenance and Reliability"},
      {"level": "h2", "text": "Buying Guide"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if profile == "security_escort":
    return [
      {"level": "h1", "text": _h1()},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": "What Is a Security Escort?"},
      {"level": "h2", "text": "When Escorts Are Required"},
      {"level": "h2", "text": "Planning and Risk Assessment"},
      {"level": "h2", "text": "Protocols and Best Practices"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if intent == "commercial":
    return [
      {"level": "h1", "text": _h1()},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"What Is {primary.title()}?"},
      {"level": "h2", "text": "Key Options Compared"},
      {"level": "h2", "text": "How to Choose the Right Fit"},
      {"level": "h2", "text": "Pricing and Value Factors"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if intent == "informational":
    return [
      {"level": "h1", "text": _h1()},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"What Is {primary.title()}?"},
      {"level": "h2", "text": "How It Works"},
      {"level": "h2", "text": "Benefits and Use Cases"},
      {"level": "h2", "text": "Step-by-Step Guide"},
      {"level": "h2", "text": "Conclusion"},
    ]
  return [
    {"level": "h1", "text": _h1()},
    {"level": "h2", "text": "Introduction"},
    {"level": "h2", "text": f"Understanding {primary.title()}"},
    {"level": "h2", "text": "Key Considerations"},
    {"level": "h2", "text": "Practical Guidance"},
    {"level": "h2", "text": "Conclusion"},
  ]


_PAA_TEMPLATES = (
  "What is {primary}?",
  "How does {primary} work?",
  "Is {primary} safe?",
  "How much does {primary} cost?",
  "What should I know before choosing {primary}?",
  "Who typically uses {primary}?",
)


def generate_paa_faqs(
  topic: str,
  primary: str,
  facts: list[StrictFact],
  profile: str,
  intent: str,
  *,
  keywords: list[str] | None = None,
  domain: str = "general",
  locations: list[str] | None = None,
  seed: int = 0,
) -> list[dict[str, str]]:
  from app.engine.seo_content_enrichment import (
    classify_intents_extended,
    generate_unique_faqs,
  )

  locs = locations or []
  intents = classify_intents_extended(topic, keywords or [], locs, intent)
  return generate_unique_faqs(
    topic,
    primary,
    keywords or [],
    facts,
    profile=profile,
    domain=domain,
    intents=intents,
    locations=locs,
    seed=seed,
  )


def generate_safe_metadata(
  title: str,
  article: str,
  primary: str,
  topic: str,
) -> str:
  """Meta description from article only — never polluted retrieval chunks."""
  intro = ""
  for line in article.splitlines():
    line = line.strip()
    if not line or line.startswith("#") or line.startswith(">"):
      continue
    if line.startswith("|") or line.startswith("---"):
      continue
    intro = line
    break
  if not intro:
    intro = f"A practical guide to {primary} covering {topic}."
  intro = re.sub(r"\*+", "", intro).strip()
  if len(intro) > 130:
    intro = intro[:127].rsplit(" ", 1)[0] + "..."
  meta = f"{intro} Learn key considerations, options, and answers to common questions."
  return meta[:160].rsplit(" ", 1)[0] + "..." if len(meta) > 160 else meta


def build_profile_article_from_outline(
  topic: str,
  primary: str,
  outline: list[dict[str, str]],
  *,
  profile: str,
  intent: str,
) -> str:
  """Template article from dynamic outline — no polluted RAG facts."""
  sections: list[str] = []
  title = outline[0]["text"] if outline else f"{topic} Guide"
  sections.extend([f"# {title}", "", "## Introduction", ""])

  if profile == "adult_services":
    sections.append(
      f"This guide covers **{topic}** with a focus on safety, privacy, agency selection, "
      "booking etiquette, and pricing transparency. Information is presented professionally "
      "for readers researching companion services."
    )
  else:
    sections.append(
      f"This guide explains **{primary}** in the context of **{topic}** with practical, "
      "structured information aligned to your search intent."
    )
  sections.append("")

  section_bodies = {
    "Understanding Escort Services": (
      "Escort services typically involve pre-arranged companionship. Reputable providers "
      "emphasize consent, discretion, clear communication, and professional conduct."
    ),
    "Safety and Privacy Considerations": (
      "Verify agency credentials, avoid sharing sensitive personal data prematurely, "
      "meet in safe locations, and confirm policies on confidentiality before booking."
    ),
    "Types of Companion Services": (
      "Services may include social companionship, event attendance, travel companionship, "
      "and other arrangements depending on local regulations and agency policies."
    ),
    "Choosing Reputable Agencies": (
      "Look for established agencies with verified reviews, transparent pricing, "
      "professional websites, and responsive customer support."
    ),
    "Booking Process": (
      "Typical steps include initial inquiry, availability confirmation, service selection, "
      "deposit or payment terms, and confirmation of meeting details."
    ),
    "Pricing Factors": (
      "Pricing may vary by duration, service type, location, and provider experience. "
      "Request a clear quote before confirming."
    ),
    "Frequently Asked Questions": (
      "See the FAQ section below for common questions about safety, booking, and expectations."
    ),
  }

  for item in outline:
    heading = item.get("text", "")
    if item.get("level") != "h2" or heading.lower() in ("introduction", "conclusion"):
      continue
    sections.extend([f"## {heading}", ""])
    body = section_bodies.get(
      heading,
      f"When researching {heading.lower()}, prioritize verified information, safety, "
      f"and policies relevant to {topic}.",
    )
    sections.append(body)
    sections.append("")

  sections.extend([
    "## Conclusion",
    "",
    f"Research **{topic}** carefully, prioritize safety and discretion, and choose "
    "reputable providers with transparent policies.",
  ])
  return re.sub(r"\n{3,}", "\n\n", "\n".join(sections)).strip()


def polluted_fact(text: str, anchors: list[str], *, topic: str = "", keywords: list[str] | None = None) -> bool:
  if not anchor_match_count(text, anchors):
    return True
  low = text.lower()
  noise = ("mastercard", "ajay banga", "executive chairman", "world bank president")
  if any(n in low for n in noise):
    return True
  blob = f"{topic} {' '.join(keywords or [])}".lower()
  if "flutter" in blob and any(w in blob for w in ("state", "widget", "dart", "riverpod", "bloc")):
    if any(g in low for g in ("everymatrix", "online gambling", "online casino", "betting", "sportsbook")):
      return True
  return False
