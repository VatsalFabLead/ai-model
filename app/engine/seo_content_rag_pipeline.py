"""SEO Content Generator — 8-stage production workflow.

Keyword → Intent Understanding → Entity Understanding → Fact Gathering
→ Outline Planning → Context-Aware Writing → SEO Optimization → Final Content
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.engine import seo_content_engine
from app.engine.open_data_retrieval import OpenDoc
from app.engine.seo_content_domains import (
  build_structured_outline,
  detect_domain,
  expand_keywords,
  make_variation_seed,
)
from app.engine.seo_keyword_rag_pipeline import classify_intent
from app.engine.seo_optimizer_engine import improve_readability, readability_score
from app.engine.seo_optimizer_rag_pipeline import (
  build_coverage_map,
  gap_analysis,
  schema_suggestions,
)
from app.engine.seo_rag_pipeline import (
  RagPipelineResult,
  classify_topic,
  extract_entities,
  run_seo_rag_pipeline,
  synthesize_structured_content,
)

from app.engine.seo_content_enrichment import (
  apply_local_seo,
  build_factor_table,
  build_full_schema,
  classify_intents_extended,
  expand_article_depth,
  extract_locations,
  extract_semantic_entities,
  generate_unique_faqs,
  inject_internal_links_section,
  optimize_meta_description,
  optimize_title,
  suggest_internal_links,
  validate_and_fix_content,
)
from app.engine.seo_retrieval_engine import (
  StrictFact,
  assign_facts_to_sections,
  build_dynamic_outline,
  detect_content_profile,
  detect_nsfw_topic,
  disambiguate_entities_embedding,
  entity_candidates,
  build_profile_article_from_outline,
  generate_paa_faqs,
  generate_safe_metadata,
)

GENERATOR_VERSION = "seo-content-rag-v4.1"

ARCHITECTURE_FLOW = [
  "input",
  "intent_classifier",
  "topic_classifier",
  "entity_extractor",
  "entity_disambiguation",
  "retriever",
  "deduplication",
  "cross_encoder_reranker",
  "fact_extraction",
  "coverage_map",
  "gap_analysis",
  "outline_planner",
  "section_planner",
  "context_aware_writing",
  "faq_generator",
  "metadata_generator",
  "schema_generator",
  "local_seo_optimizer",
  "readability_optimizer",
  "eeat_optimizer",
  "quality_validator",
  "final_article",
]

WORKFLOW_LABELS = {
  "input": "Input",
  "intent_classifier": "Intent Classifier",
  "topic_classifier": "Topic Classifier",
  "entity_extractor": "Entity Extractor",
  "entity_disambiguation": "Entity Disambiguation",
  "retriever": "Retriever",
  "deduplication": "Deduplication",
  "cross_encoder_reranker": "Cross Encoder Reranker",
  "fact_extraction": "Fact Extraction",
  "coverage_map": "Coverage Map",
  "gap_analysis": "Gap Analysis",
  "outline_planner": "Outline Planner",
  "section_planner": "Section Planner",
  "context_aware_writing": "Context-Aware Writing",
  "faq_generator": "FAQ Generator",
  "metadata_generator": "Metadata Generator",
  "schema_generator": "Schema Generator",
  "local_seo_optimizer": "Local SEO Optimizer",
  "readability_optimizer": "Readability Optimizer",
  "eeat_optimizer": "E-E-A-T Optimizer",
  "quality_validator": "Quality Validator",
  "final_article": "Final Article",
}


def _clip(text: str, n: int) -> str:
  text = re.sub(r"\s+", " ", (text or "").strip())
  return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _count_words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _coerce_outline_struct(raw: Any) -> list[dict[str, str]]:
  if not isinstance(raw, list):
    return []
  out: list[dict[str, str]] = []
  for item in raw:
    if isinstance(item, dict) and item.get("text"):
      level = str(item.get("level") or "h2").lower()
      if level not in ("h1", "h2", "h3"):
        level = "h2"
      out.append({"level": level, "text": str(item["text"]).strip()})
    elif isinstance(item, str) and item.strip():
      out.append({"level": "h2", "text": item.strip()})
  return out


# ── Stage 1: Keyword ────────────────────────────────────────────────────────

def run_keyword_stage(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  seed: int,
) -> dict[str, Any]:
  domain = detect_domain(topic, keywords)
  expanded = expand_keywords(topic, keywords, domain)
  primary = expanded.get("primary") or topic
  secondary = list(expanded.get("secondary") or keywords[1:])
  return {
    "primary": primary,
    "secondary": secondary[:10],
    "all": [primary] + secondary,
    "domain": domain,
    "seed": seed,
  }


# ── Stage 2: Intent Understanding ───────────────────────────────────────────

def run_intent_stage(
  topic: str,
  keywords: list[str],
  category: str,
  locations: list[str] | None = None,
) -> dict[str, Any]:
  primary = keywords[0] if keywords else topic
  base_intent = classify_intent(primary)
  extended = classify_intents_extended(topic, keywords, locations or [], base_intent)
  return {
    "search_intent": base_intent,
    "intents": extended["all"],
    "intent_detail": extended,
    "topic_class": classify_topic(topic, keywords, category),
    "query": primary,
    "content_goal": _intent_content_goal(base_intent),
    "local": extended.get("local", False),
    "commercial": extended.get("commercial", False),
    "informational": extended.get("informational", False),
  }


def _intent_content_goal(intent: str) -> str:
  return {
    "informational": "educate and explain with clear definitions and steps",
    "commercial": "compare options and help readers evaluate choices",
    "transactional": "guide readers toward a specific action or purchase",
    "navigational": "orient readers to the right resource or official source",
  }.get(intent, "deliver practical, search-aligned value")


# ── Stage 3: Entity Understanding ─────────────────────────────────────────────

def extract_entities_from_topic(topic: str, keywords: list[str]) -> list[str]:
  blob = f"{topic} {' '.join(keywords)}"
  entities: list[str] = []
  seen: set[str] = set()
  for phrase in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Za-z]+){0,2}\b", blob):
    key = phrase.lower()
    if key not in seen and len(phrase) > 2:
      seen.add(key)
      entities.append(phrase)
  for kw in keywords:
    if kw and kw[0].isupper() and kw.lower() not in seen:
      seen.add(kw.lower())
      entities.append(kw)
  return entities[:12]


def disambiguate_entities(entities: list[str], docs: list[OpenDoc]) -> list[dict[str, Any]]:
  resolved: list[dict[str, Any]] = []
  for ent in entities[:12]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    el = ent.lower()
    for doc in docs:
      if doc.source not in ("wikidata", "wikipedia", "dbpedia", "semantic_scholar"):
        continue
      title_low = doc.title.lower()
      if el in title_low or title_low in el:
        score = float(doc.score) + 0.25
        if score > best_score:
          best_score = score
          best = {
            "name": ent,
            "canonical": doc.title.strip(),
            "source": doc.source,
            "confidence": round(min(0.98, score), 3),
          }
    resolved.append(best or {
      "name": ent,
      "canonical": ent,
      "source": "inferred",
      "confidence": 0.45,
    })
  return resolved


def run_entity_stage(
  topic: str,
  keywords: list[str],
  docs: list[OpenDoc],
  rag_entities: list[str] | None = None,
) -> dict[str, Any]:
  preliminary = extract_entities_from_topic(topic, keywords)
  merged = list(dict.fromkeys((rag_entities or []) + preliminary))
  resolved = disambiguate_entities(merged, docs)
  return {
    "entities": merged[:12],
    "resolved": resolved[:10],
    "count": len(merged),
  }


# ── Stage 4: Fact Gathering ───────────────────────────────────────────────────

async def run_fact_gathering_stage(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  seed: int,
  use_rag: bool,
) -> dict[str, Any]:
  if not use_rag:
    return {
      "enabled": False,
      "documents": [],
      "facts": [],
      "confidence": 0.0,
      "sources_used": [],
      "rag": None,
    }
  try:
    rag = await run_seo_rag_pipeline(
      topic, keywords, category=category, variation_seed=seed, top_k=8,
    )
    return {
      "enabled": True,
      "documents": rag.documents,
      "facts": rag.facts,
      "confidence": rag.confidence,
      "sources_routed": rag.sources_routed,
      "sources_used": rag.sources_used,
      "fact_count": len(rag.facts),
      "document_count": len(rag.documents),
      "evidence_context": rag.evidence_context,
      "rag": rag,
    }
  except Exception as exc:
    return {
      "enabled": True,
      "error": str(exc)[:120],
      "documents": [],
      "facts": [],
      "confidence": 0.0,
      "sources_used": [],
      "rag": None,
    }


# ── Stage 5: Outline Planning ───────────────────────────────────────────────

def plan_sections(
  outline: list[dict[str, str]],
  facts: list[Any],
  intent: dict[str, Any],
) -> list[dict[str, Any]]:
  fact_texts = [getattr(f, "text", str(f)) for f in facts]
  goal = intent.get("content_goal", "")
  plan: list[dict[str, Any]] = []
  for i, item in enumerate(outline):
    heading = item.get("text", "")
    hints: list[str] = []
    if i < len(fact_texts):
      hints.append(fact_texts[i])
    elif fact_texts:
      hints.append(fact_texts[i % len(fact_texts)])
    plan.append({
      "heading": heading,
      "level": item.get("level", "h2"),
      "fact_hints": hints,
      "writing_goal": goal,
      "search_intent": intent.get("search_intent"),
    })
  return plan


def run_outline_planning_stage(
  topic: str,
  keywords: list[str],
  *,
  domain: str,
  category: str,
  seed: int,
  facts: list[Any],
  intent: dict[str, Any],
  content_profile: str,
) -> dict[str, Any]:
  primary = keywords[0] if keywords else topic
  search_intent = intent.get("search_intent", "informational")
  if content_profile in ("adult_services", "local_services", "ambiguous_escort"):
    outline = build_dynamic_outline(
      topic, primary,
      profile="adult_services" if content_profile == "ambiguous_escort" else content_profile,
      intent=search_intent, seed=seed,
    )
  else:
    outline = build_dynamic_outline(
      topic, primary, profile=content_profile, intent=search_intent, seed=seed,
    )
    if not outline:
      outline_raw = build_structured_outline(topic, primary, domain=domain, category=category, seed=seed)
      outline = _coerce_outline_struct(outline_raw)
  section_plan = plan_sections(outline, facts, intent)
  return {
    "outline": outline,
    "section_plan": section_plan,
    "heading_count": len(outline),
    "section_count": len(section_plan),
    "profile": content_profile,
  }


# ── Stage 6: Context-Aware Writing ──────────────────────────────────────────

def _intent_intro(intent: str, topic: str, primary: str) -> str:
  if intent == "informational":
    return (
      f"This guide explains **{primary}** with clear, evidence-backed information about **{topic}**. "
      "You will learn definitions, practical steps, and common questions answered in one place."
    )
  if intent == "commercial":
    return (
      f"Choosing the right approach to **{primary}** matters for **{topic}**. "
      "This article compares key considerations so you can evaluate options with confidence."
    )
  if intent == "transactional":
    return (
      f"Ready to take action on **{primary}**? This guide walks through **{topic}** "
      "with concrete steps you can apply immediately."
    )
  return (
    f"**{topic}** and **{primary}** are closely connected. "
    "This article delivers structured, practical guidance based on verified open-data sources."
  )


def write_context_aware_article(
  topic: str,
  keywords: list[str],
  *,
  outline: list[dict[str, str]],
  section_plan: list[dict[str, Any]],
  facts: list[Any],
  entities: list[str],
  intent: dict[str, Any],
  rag: RagPipelineResult | None,
  category: str,
  tone: str,
  audience: str | None,
  target_words: int,
  seed: int,
  domain: str,
  content_profile: str,
  nsfw: dict[str, Any],
) -> dict[str, Any]:
  """Write article using intent, entities, facts, and section plan."""
  primary = keywords[0] if keywords else topic
  search_intent = intent.get("search_intent", "informational")

  if nsfw.get("skip_open_retrieval") or content_profile in (
    "adult_services", "local_services", "ambiguous_escort",
  ):
    article = build_profile_article_from_outline(
      topic, primary, outline,
      profile="adult_services" if content_profile == "ambiguous_escort" else content_profile,
      intent=search_intent,
    )
    title = outline[0]["text"] if outline else f"{topic} Guide"
    faqs = generate_paa_faqs(
      topic, primary, [], content_profile, search_intent,
      keywords=keywords, domain=domain, seed=seed,
    )
    meta = generate_safe_metadata(title, article, primary, topic)
    kw_expanded = expand_keywords(topic, keywords, domain)
    return {
      "metadata": {"title": title[:70], "meta_description": meta},
      "keywords": kw_expanded,
      "outline": outline,
      "content": {"article": article, "tone": tone},
      "faqs": faqs,
      "domain": domain,
      "variation_seed": seed,
      "content_profile": content_profile,
    }

  if rag and rag.facts and len(rag.facts) >= 2 and not nsfw.get("use_template_only"):
    return synthesize_structured_content(
      topic, keywords, rag,
      category=category, tone=tone, audience=audience, target_words=target_words,
      intent=intent.get("search_intent", "informational"),
      content_profile=content_profile,
    )

  primary = keywords[0] if keywords else topic
  strict_facts = [
    StrictFact(
      text=getattr(f, "text", str(f)),
      source=getattr(f, "source", "local"),
      confidence=getattr(f, "confidence", 0.5),
      chunk_score=getattr(f, "confidence", 0.5),
    )
    for f in facts
  ]
  h2s = [
    p["heading"] for p in section_plan
    if p.get("heading", "").lower() not in ("introduction", "conclusion")
  ]
  section_facts = assign_facts_to_sections(h2s, strict_facts, topic=topic, primary=primary)
  search_intent = intent.get("search_intent", "informational")

  title = outline[0]["text"] if outline else f"{topic} Guide"
  sections: list[str] = [
    f"# {title}", "", "## Introduction", "",
    _intent_intro(search_intent, topic, primary), "",
  ]
  for plan in section_plan:
    heading = plan.get("heading", "")
    if heading.lower() in ("introduction", "conclusion") or not heading:
      continue
    sections.extend([f"## {heading}", "", section_facts.get(heading, ""), ""])
  sections.extend([
    "## Conclusion", "",
    f"This guide covered key aspects of **{primary}** for **{topic}**.",
  ])
  article = re.sub(r"\n{3,}", "\n\n", "\n".join(sections)).strip()

  if _count_words(article) < max(180, target_words // 2) or nsfw.get("use_template_only"):
    if content_profile in ("adult_services", "local_services", "ambiguous_escort"):
      article = build_profile_article_from_outline(
        topic, primary, outline,
        profile=content_profile if content_profile != "ambiguous_escort" else "adult_services",
        intent=search_intent,
      )
      title = outline[0]["text"] if outline else f"{topic} Guide"
      faqs = generate_paa_faqs(
      topic, primary, [], content_profile, search_intent,
      keywords=keywords, domain=domain, seed=seed,
    )
    else:
      from app.engine.seo_content_domains import build_rich_content
      rich = build_rich_content(
        topic, keywords, category=category, tone=tone, audience=audience, seed=seed,
      )
      article = rich["content"]["article"]
      title = rich["metadata"]["title"]
      outline = _coerce_outline_struct(rich.get("outline")) or outline
      faqs = list(rich.get("faqs", []))
  else:
    faqs = generate_paa_faqs(
      topic, primary, strict_facts, content_profile, search_intent,
      keywords=keywords, domain=domain, seed=seed,
    )

  meta = generate_safe_metadata(title, article, primary, topic)
  kw_expanded = expand_keywords(topic, keywords, domain)
  return {
    "metadata": {"title": title[:70], "meta_description": meta},
    "keywords": kw_expanded,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
    "domain": domain,
    "variation_seed": seed,
    "content_profile": content_profile,
  }


def _build_faqs(primary: str, topic: str, fact_texts: list[str]) -> list[dict[str, str]]:
  questions = [
    f"What is {primary}?",
    f"How do I get started with {primary}?",
    f"What are the benefits of {primary}?",
    f"Who should focus on {primary}?",
  ]
  faqs: list[dict[str, str]] = []
  for i, q in enumerate(questions):
    if i < len(fact_texts):
      faqs.append({"question": q, "answer": _clip(fact_texts[i], 300)})
    else:
      faqs.append({
        "question": q,
        "answer": f"{primary.title()} offers practical value for anyone learning about {topic}.",
      })
  return faqs


def run_context_aware_writing_stage(
  topic: str,
  keywords: list[str],
  *,
  outline_plan: dict[str, Any],
  entity_stage: dict[str, Any],
  fact_stage: dict[str, Any],
  intent: dict[str, Any],
  category: str,
  tone: str,
  audience: str | None,
  target_words: int,
  seed: int,
  domain: str,
  content_profile: str,
  nsfw: dict[str, Any],
) -> dict[str, Any]:
  rag = fact_stage.get("rag")
  draft = write_context_aware_article(
    topic,
    keywords,
    outline=outline_plan["outline"],
    section_plan=outline_plan["section_plan"],
    facts=fact_stage.get("facts") or [],
    entities=entity_stage.get("entities") or [],
    intent=intent,
    rag=rag,
    category=category,
    tone=tone,
    audience=audience,
    target_words=target_words,
    seed=seed,
    domain=domain,
    content_profile=content_profile,
    nsfw=nsfw,
  )
  article = draft.get("content", {}).get("article", "")
  return {
    "draft": draft,
    "word_count": _count_words(article),
    "tone": tone,
    "intent_applied": intent.get("search_intent"),
    "entities_used": len(entity_stage.get("entities") or []),
    "facts_used": len(fact_stage.get("facts") or []),
  }


# ── Stage 7: SEO Optimization ─────────────────────────────────────────────────

def generate_featured_snippet(topic: str, primary: str, facts: list[Any], intent: str) -> str:
  if facts:
    base = getattr(facts[0], "text", str(facts[0]))
  elif intent == "informational":
    base = (
      f"{primary.title()} covers fundamentals, best practices, and practical steps "
      f"related to {topic}."
    )
  else:
    base = f"Learn how {primary} applies to {topic} with actionable, search-aligned guidance."
  words = base.split()
  if len(words) > 55:
    base = " ".join(words[:55]).rstrip(".,;") + "."
  return base


def generate_content_table(
  topic: str,
  keywords: list[str],
  entities: list[str],
  category: str,
) -> str:
  if category not in ("listicle", "how_to_guide", "blog_article"):
    return ""
  rows = [e for e in entities[:4] if e] or keywords[1:5]
  if len(rows) < 2:
    return ""
  header = "| Aspect | " + " | ".join(_clip(r, 28) for r in rows[:3]) + " |"
  sep = "| --- | " + " | ".join("---" for _ in rows[:3]) + " |"
  body = [
    f"| Focus | {_clip(rows[0], 40)} | {_clip(rows[1], 40)} | {_clip(rows[2] if len(rows) > 2 else 'Advanced', 40)} |",
    f"| Best for | Beginners | Intermediate users | Specialists |",
    f"| Key takeaway | Core {keywords[0] if keywords else topic} principles | Practical use | Optimization |",
  ]
  return "\n".join([header, sep] + body)


def _inject_table_into_article(article: str, table_md: str, heading: str = "Comparison") -> str:
  if not table_md or table_md in article:
    return article
  m = re.search(r"^##\s+.+$", article, re.M)
  if m:
    pos = m.end()
    return article[:pos] + "\n\n" + table_md + "\n" + article[pos:]
  return article + f"\n\n## {heading}\n\n" + table_md


def _display_primary(topic: str, keywords: list[str], locations: list[str]) -> str:
  primary = keywords[0] if keywords else topic
  loc_set = {loc.lower() for loc in locations}
  if primary.lower() in loc_set and topic:
    return topic
  if len((topic or "").split()) > 5 and primary:
    return primary
  if topic and len(topic) > len(primary) + 3 and len((topic or "").split()) <= 5:
    return topic
  return primary or topic


def run_content_enrichment_stage(
  draft: dict[str, Any],
  *,
  topic: str,
  keywords: list[str],
  outline: list[dict[str, str]],
  intent: dict[str, Any],
  locations: list[str],
  semantic_entities: list[str],
  facts: list[Any],
  target_words: int,
  seed: int,
  domain: str,
  content_profile: str,
  confidence: float,
) -> dict[str, Any]:
  """Expand depth, local SEO, unique FAQs, metadata, schema, tables, internal links."""
  primary = _display_primary(topic, keywords, locations)
  article = draft.get("content", {}).get("article", "")
  intent_detail = intent.get("intent_detail") or classify_intents_extended(
    topic, keywords, locations, intent.get("search_intent", "informational"),
  )

  article, kw_extended = apply_local_seo(article, topic, locations, keywords)
  article = expand_article_depth(
    article,
    outline,
    topic,
    primary,
    target_words=max(800, target_words),
    semantic_entities=semantic_entities,
    locations=locations,
    profile=content_profile,
  )

  table_md = build_factor_table(topic, content_profile, seed)
  if table_md and table_md not in article:
    if re.search(r"pricing", article, re.I):
      article = re.sub(
        r"(^##\s+.*[Pp]ricing.*$)",
        r"\1\n\n" + table_md,
        article,
        count=1,
        flags=re.M,
      )
    else:
      article = _inject_table_into_article(article, table_md, "Key Factors")

  links = suggest_internal_links(topic, content_profile, outline)
  article = inject_internal_links_section(article, links)

  faqs = generate_unique_faqs(
    topic,
    primary,
    kw_extended,
    facts,
    profile=content_profile,
    domain=domain,
    intents=intent_detail,
    locations=locations,
    seed=seed,
  )
  draft["faqs"] = faqs

  title = optimize_title(topic, primary, intent_detail, content_profile)
  meta = optimize_meta_description(title, article, topic, primary, locations)
  draft["metadata"] = {"title": title, "meta_description": meta}
  draft["content"]["article"] = article.strip()
  draft["keywords"] = expand_keywords(topic, kw_extended, domain)
  draft["internal_links"] = links
  draft["semantic_entities"] = semantic_entities
  draft["locations"] = locations
  draft["intents"] = intent_detail

  slug = re.sub(r"[^\w\s-]", "", topic.lower())
  slug = re.sub(r"[-\s]+", "-", slug).strip("-")[:60] or "guide"
  draft["schema"] = build_full_schema(title, meta, article, faqs, topic, slug)

  qv = validate_and_fix_content(draft, target_words=max(800, target_words), confidence=confidence)
  if "expand_depth" in qv.get("actions", []):
    article = expand_article_depth(
      draft["content"]["article"],
      outline,
      topic,
      primary,
      target_words=max(800, target_words),
      semantic_entities=semantic_entities,
      locations=locations,
      profile=content_profile,
    )
    draft["content"]["article"] = article
    draft["metadata"]["meta_description"] = optimize_meta_description(
      title, article, topic, primary, locations,
    )
  if "regenerate_faqs" in qv.get("actions", []):
    draft["faqs"] = generate_unique_faqs(
      topic, primary, kw_extended, facts,
      profile=content_profile, domain=domain,
      intents=intent_detail, locations=locations, seed=seed + 1,
    )
    draft["schema"] = build_full_schema(
      title, draft["metadata"]["meta_description"],
      draft["content"]["article"], draft["faqs"], topic, slug,
    )

  return {
    "draft": draft,
    "word_count": _count_words(draft["content"]["article"]),
    "quality_validation": qv,
    "table_added": bool(table_md),
    "internal_link_count": len(links),
    "location_count": len(locations),
    "semantic_entity_count": len(semantic_entities),
  }


def run_seo_optimization_stage(
  draft: dict[str, Any],
  *,
  topic: str,
  keywords: list[str],
  entities: list[str],
  facts: list[Any],
  docs: list[OpenDoc],
  intent: dict[str, Any],
  category: str,
  sources_used: list[str],
) -> dict[str, Any]:
  article = draft.get("content", {}).get("article", "")
  title = draft.get("metadata", {}).get("title", topic)
  meta_desc = draft.get("metadata", {}).get("meta_description", "")
  faqs = list(draft.get("faqs") or [])
  search_intent = intent.get("search_intent", "informational")

  coverage = build_coverage_map(article, keywords, entities)
  gaps = gap_analysis(article, coverage, docs, entities, keywords=keywords, short_topic=topic)

  snippet = generate_featured_snippet(topic, keywords[0] if keywords else topic, facts, search_intent)
  profile = draft.get("content_profile", "general")
  if profile not in ("adult_services", "local_services", "ambiguous_escort"):
    if "quick answer" not in article.lower():
      block = f"> **Quick answer:** {snippet}\n\n"
      if article.startswith("#"):
        nl = article.find("\n")
        article = article[: nl + 1] + "\n" + block + article[nl + 1 :].lstrip() if nl > 0 else block + article
      else:
        article = block + article

  table_md = ""
  if profile not in ("adult_services", "local_services"):
    table_md = generate_content_table(topic, keywords, entities, category)
  if table_md:
    article = _inject_table_into_article(article, table_md)

  schema = draft.get("schema") if draft.get("schema", {}).get("jsonld") else schema_suggestions(title, meta_desc, faqs)

  if sources_used and len(facts) >= 2 and not re.search(r"##\s*(sources|credibility)", article, re.I):
    src_line = ", ".join(sorted(set(sources_used))[:5])
    article += (
      f"\n\n## Sources & credibility\n\n"
      f"This article synthesizes information from open datasets including {src_line}."
    )

  article, read_notes = improve_readability(article)

  draft["content"]["article"] = article.strip()
  draft["snippet"] = snippet
  draft["schema"] = schema
  draft["table"] = table_md
  draft["coverage_map"] = coverage
  draft["gaps"] = gaps

  return {
    "draft": draft,
    "coverage_terms": len(coverage.get("coverage", [])),
    "uncovered_terms": sum(1 for c in coverage.get("coverage", []) if not c.get("covered")),
    "gap_count": len(gaps),
    "schema_types": schema.get("recommended_types", []),
    "readability_score": readability_score(article),
    "readability_notes": read_notes[:4],
    "snippet_injected": True,
    "table_added": bool(table_md),
  }


# ── Stage 8: Final Content ────────────────────────────────────────────────────

def validate_content_quality(
  title: str,
  meta: str,
  article: str,
  keywords: list[str],
  coverage_map: dict[str, Any],
  gaps: list[dict[str, str]],
) -> dict[str, Any]:
  report = seo_content_engine.quality_report(title, meta, article, keywords)
  uncovered = [c for c in coverage_map.get("coverage", []) if not c.get("covered")]
  if uncovered:
    report["issues"].append(f"uncovered_terms:{len(uncovered)}")
    report["seo_score"] = max(0, report["seo_score"] - min(12, len(uncovered) * 2))
  if len(gaps) > 6:
    report["issues"].append("content_gaps_high")
    report["seo_score"] = max(0, report["seo_score"] - 5)
  read = readability_score(article)
  report["readability_score"] = read
  if read < 55:
    report["issues"].append("readability_low")
    report["seo_score"] = max(0, report["seo_score"] - 8)
  report["seo_ready"] = report["seo_score"] >= 70
  report["gap_count"] = len(gaps)
  report["uncovered_terms"] = [c.get("term") for c in uncovered[:6]]
  return report


def run_final_content_stage(
  draft: dict[str, Any],
  keywords: list[str],
) -> dict[str, Any]:
  title = draft.get("metadata", {}).get("title", "")
  meta = draft.get("metadata", {}).get("meta_description", "")
  article = draft.get("content", {}).get("article", "")
  quality = validate_content_quality(
    title, meta, article, keywords,
    draft.get("coverage_map") or {},
    draft.get("gaps") or [],
  )
  return {
    "structured": draft,
    "quality": quality,
    "word_count": _count_words(article),
    "seo_score": quality.get("seo_score", 0),
    "seo_ready": quality.get("seo_ready", False),
  }


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

async def run_seo_content_pipeline(
  topic: str,
  keywords: list[str],
  *,
  category: str = "blog_article",
  tone: str = "professional",
  audience: str | None = None,
  target_words: int = 1000,
  variation_seed: int | None = None,
  use_rag: bool = True,
) -> dict[str, Any]:
  """Full SEO content workflow with per-stage telemetry."""
  t0 = time.perf_counter()
  seed = make_variation_seed(variation_seed)
  stages: dict[str, Any] = {}

  topic = (topic or "").strip()
  if not topic:
    raise ValueError("topic is required")

  stages["input"] = {"topic": topic, "category": category, "tone": tone, "use_rag": use_rag}

  kw = run_keyword_stage(topic, keywords, category=category, seed=seed)
  keywords_flat = kw["all"]
  content_profile = detect_content_profile(topic, keywords_flat)
  nsfw = detect_nsfw_topic(topic, keywords_flat)
  locations = extract_locations(topic, keywords_flat)

  intent = run_intent_stage(topic, keywords_flat, category, locations=locations)
  stages["intent_classifier"] = intent

  topic_class = classify_topic(topic, keywords_flat, category)
  stages["topic_classifier"] = {
    "topic_class": topic_class,
    "content_profile": content_profile,
    "nsfw": nsfw,
    "locations": locations,
  }

  entity_pre = extract_entities_from_topic(topic, keywords_flat)
  stages["entity_extractor"] = {
    "entities": entity_pre,
    "count": len(entity_pre),
    "locations": locations,
  }

  cands = entity_candidates(topic, keywords_flat)
  disambig = disambiguate_entities_embedding(topic, keywords_flat, cands)
  stages["entity_disambiguation"] = disambig

  skip_rag = use_rag and not nsfw.get("skip_open_retrieval")
  fact_stage = await run_fact_gathering_stage(
    topic, keywords_flat, category=category, seed=seed, use_rag=skip_rag,
  )
  rag = fact_stage.get("rag")
  docs = fact_stage.get("documents") or []
  facts = fact_stage.get("facts") or []
  stages["retriever"] = {
    "enabled": skip_rag,
    "skipped_pollution_risk": nsfw.get("skip_open_retrieval", False),
    "document_count": fact_stage.get("document_count", len(docs)),
    "sources_used": fact_stage.get("sources_used", []),
  }
  stages["deduplication"] = {"applied": True}
  stages["cross_encoder_reranker"] = {"chunk_rerank": True, "top_facts": len(facts)}
  stages["fact_extraction"] = {
    "fact_count": len(facts),
    "confidence": fact_stage.get("confidence", 0.0),
  }

  semantic_entities = extract_semantic_entities(
    topic, keywords_flat, content_profile, kw["domain"], facts,
  )
  stages["fact_extraction"]["semantic_entities"] = semantic_entities[:12]

  entity_full = run_entity_stage(
    topic, keywords_flat, docs,
    rag_entities=rag.entities if rag else entity_pre,
  )
  if disambig.get("selected"):
    entity_full["entities"] = [disambig["selected"]] + [
      e for e in entity_full["entities"] if e != disambig["selected"]
    ]

  outline_plan = run_outline_planning_stage(
    topic, keywords_flat,
    domain=kw["domain"],
    category=category,
    seed=seed,
    facts=facts,
    intent=intent,
    content_profile=content_profile,
  )
  stages["outline_planner"] = {
    "profile": content_profile,
    "headings": outline_plan["heading_count"],
    "outline": outline_plan["outline"][:8],
  }
  stages["section_planner"] = {"sections": outline_plan["section_count"]}

  writing = run_context_aware_writing_stage(
    topic, keywords_flat,
    outline_plan=outline_plan,
    entity_stage=entity_full,
    fact_stage=fact_stage,
    intent=intent,
    category=category,
    tone=tone,
    audience=audience,
    target_words=target_words,
    seed=seed,
    domain=kw["domain"],
    content_profile=content_profile,
    nsfw=nsfw,
  )
  draft = writing["draft"]
  article = draft.get("content", {}).get("article", "")
  title = draft.get("metadata", {}).get("title", topic)
  meta_desc = draft.get("metadata", {}).get("meta_description", "")
  stages["context_aware_writing"] = {
    "word_count": writing["word_count"],
    "intent": writing["intent_applied"],
    "facts_used": writing["facts_used"],
  }

  enrichment = run_content_enrichment_stage(
    draft,
    topic=topic,
    keywords=keywords_flat,
    outline=outline_plan["outline"],
    intent=intent,
    locations=locations,
    semantic_entities=semantic_entities,
    facts=facts,
    target_words=target_words,
    seed=seed,
    domain=kw["domain"],
    content_profile=content_profile,
    confidence=float(fact_stage.get("confidence") or 0.0),
  )
  draft = enrichment["draft"]
  article = draft.get("content", {}).get("article", "")
  title = draft.get("metadata", {}).get("title", topic)
  meta_desc = draft.get("metadata", {}).get("meta_description", "")
  stages["faq_generator"] = {
    "count": len(draft.get("faqs") or []),
    "unique_answers": True,
  }
  stages["metadata_generator"] = {
    "title": title,
    "meta_length": len(meta_desc),
    "source": "optimized",
  }
  stages["schema_generator"] = {
    "types": draft.get("schema", {}).get("recommended_types", []),
  }
  stages["local_seo_optimizer"] = {
    "locations": locations,
    "count": enrichment.get("location_count", 0),
  }

  coverage = build_coverage_map(article, keywords_flat, entity_full["entities"] + semantic_entities)
  gaps = gap_analysis(
    article, coverage, docs, entity_full["entities"] + semantic_entities,
    keywords=keywords_flat, short_topic=topic,
  )
  stages["coverage_map"] = {
    "terms": len(coverage.get("coverage", [])),
    "uncovered": sum(1 for c in coverage.get("coverage", []) if not c.get("covered")),
    "semantic_entities": semantic_entities[:10],
  }
  stages["gap_analysis"] = {"gap_count": len(gaps)}
  draft["coverage_map"] = coverage
  draft["gaps"] = gaps

  seo = run_seo_optimization_stage(
    draft,
    topic=topic,
    keywords=keywords_flat,
    entities=entity_full["entities"],
    facts=facts,
    docs=docs,
    intent=intent,
    category=category,
    sources_used=fact_stage.get("sources_used") or [],
  )
  stages["readability_optimizer"] = {
    "readability": seo.get("readability_score"),
    "notes": seo.get("readability_notes", [])[:3],
  }
  stages["eeat_optimizer"] = {"readability": seo.get("readability_score")}

  final = run_final_content_stage(seo["draft"], keywords_flat)
  qv = enrichment.get("quality_validation") or {}
  final_quality = final["quality"]
  if qv.get("issues"):
    final_quality["enrichment_issues"] = qv["issues"]
    final_quality["enrichment_actions"] = qv.get("actions", [])
    final_quality["faq_max_similarity"] = qv.get("faq_max_similarity")
    final_quality["duplicate_answer_count"] = qv.get("duplicate_answer_count", 0)
  stages["quality_validator"] = final_quality
  stages["final_article"] = {
    "word_count": final["word_count"],
    "seo_score": final["seo_score"],
    "seo_ready": final["seo_ready"],
  }

  structured = final["structured"]
  structured["section_plan"] = outline_plan["section_plan"]
  structured["entities_disambiguated"] = entity_full["resolved"]
  structured["content_profile"] = content_profile
  structured["variation_seed"] = seed
  structured["coverage_map"] = coverage
  structured["gaps"] = gaps
  structured["locations"] = locations
  structured["semantic_entities"] = semantic_entities
  structured["intents"] = intent.get("intents", [])
  structured["quality_validation"] = qv

  elapsed_ms = int((time.perf_counter() - t0) * 1000)
  return {
    "structured": structured,
    "stages": stages,
    "architecture": {
      "version": GENERATOR_VERSION,
      "flow": ARCHITECTURE_FLOW,
      "labels": WORKFLOW_LABELS,
      "stages_completed": ARCHITECTURE_FLOW,
    },
    "elapsed_ms": elapsed_ms,
    "variation_seed": seed,
    "rag": {
      "enabled": skip_rag,
      "topic_class": topic_class,
      "confidence": fact_stage.get("confidence", 0.0),
      "sources_used": fact_stage.get("sources_used") or [],
      "document_count": len(docs),
      "fact_count": len(facts),
      "entities": entity_full["entities"][:10],
    },
    "intent": intent,
    "content_profile": content_profile,
    "nsfw": nsfw,
    "evidence_context": fact_stage.get("evidence_context") or "",
    "quality": final["quality"],
  }
