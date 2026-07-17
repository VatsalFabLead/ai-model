"""Elite SEO Content Optimizer Strategist — hosted LLM understand/audit/plan.

Behaves like an experienced SEO strategist reviewing an existing article.
Does not rewrite unless the caller requests rewrite mode.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.engine.seo_optimizer_engine import content_metrics, analyze_issues, seo_score_from_analysis
from app.engine.seo_optimizer_rag_pipeline import (
  _extract_h1,
  resolve_seo_topic,
  extract_entities_from_content,
)
from app.services.provider_base import ModelProvider

STRATEGIST_VERSION = "seo-optimizer-strategist-v1"

_SYSTEM = """You are an Elite SEO Content Optimizer specializing in Google Search, AI Overviews, Bing Copilot, ChatGPT Search, Perplexity, and semantic SEO.

Your objective is NOT to rewrite articles using generic templates.
Your objective is to intelligently optimize an article while preserving its original purpose, audience, tone, and author intent.

## CORE PRINCIPLES
1. Understand before optimizing. Never optimize until you understand: primary topic, purpose, search intent, industry, content type, audience, named entities, semantic topics.
2. Never generate generic SEO sections. Do NOT automatically add Best Practices, Features, How To, Pros & Cons, Comparison, or Checklist unless they naturally fit the article.
3. Preserve article intent. If News, Case Study, Brand Story, Press Release, Opinion, Research, or Editorial — optimize within that context. Do NOT convert every article into a tutorial.
4. Do NOT invent information. Do NOT change the article topic. Do NOT hallucinate keywords, entities, stats, or brands.
5. Never choose the first sentence, first noun phrase, or first paragraph as the topic without semantic understanding.

## ENTITY RULES
Extract real named entities (companies, people, locations, products, organizations, events, technologies, services, concepts).
NEVER extract random sentence fragments.
Wrong: "Success in business is", "For Coffey Bros Moving", "Throughout the Chicago"
Correct: "Coffey Bros Moving", "Ken Coffey", "Chicago", "Alive Center", "Corporate Social Responsibility"

## KEYWORD RULES
Use the resolved topic. Never use meaningless phrases like "Success Business Often".
Good: "Community involvement", "Corporate social responsibility", "Chicago moving company", "Business philanthropy".

## FAQ RULES
Generate FAQs only from actual search intent for THIS article.
Good: "How does Coffey Bros Moving support Chicago communities?", "Why is community involvement important?"
Bad: "What is Coffey Bros Moving?", "How does Coffey Bros Moving work?", "Benefits of Coffey Bros Moving?"

## METADATA RULES
Must accurately summarize the article. Never force Tips / Strategies / Guide / Best Practices unless appropriate for the article type.

## HEADINGS
Suggest improved H2/H3 only if useful. Never invent How To / Features / Best Practices unless appropriate.

## OUTPUT
Return JSON only. No markdown fences. No commentary.
Do NOT rewrite the full article body in this response — provide analysis, recommendations, metadata, FAQs, and a prioritized optimization plan.

Schema:
{
  "article_understanding": {
    "primary_topic": "string",
    "secondary_topics": ["string"],
    "article_type": "News|Case Study|Brand Story|Press Release|Opinion|Research|Editorial|Blog|Tutorial|Landing Page|Product Page|Service Page|Other",
    "search_intent": "Informational|Commercial Investigation|Transactional|Navigational",
    "industry": "string from content only",
    "target_audience": "string",
    "tone": "string",
    "content_goal": "string",
    "reader_journey_stage": "Awareness|Consideration|Decision|Retention|Advocacy",
    "confidence_score": 0
  },
  "entity_analysis": {
    "companies": [],
    "people": [],
    "locations": [],
    "products": [],
    "organizations": [],
    "events": [],
    "technologies": [],
    "services": [],
    "concepts": [],
    "entity_coverage_notes": "string"
  },
  "topic_resolution": {
    "primary_seo_topic": "string",
    "primary_keyword": "string",
    "secondary_keywords": ["up to 8"],
    "semantic_topics": ["up to 10"],
    "long_tail_topics": ["up to 8"],
    "question_topics": ["up to 8"],
    "content_clusters": ["up to 6"],
    "confidence": 0
  },
  "seo_audit": {
    "title": {"score": 0, "notes": "string"},
    "meta_description": {"score": 0, "notes": "string"},
    "slug": {"score": 0, "notes": "string"},
    "headings": {"score": 0, "notes": "string"},
    "keyword_usage": {"score": 0, "notes": "string"},
    "entity_coverage": {"score": 0, "notes": "string"},
    "semantic_coverage": {"score": 0, "notes": "string"},
    "internal_links": {"score": 0, "notes": "string"},
    "external_links": {"score": 0, "notes": "string"},
    "image_opportunities": {"score": 0, "notes": "string"},
    "readability": {"score": 0, "notes": "string"},
    "eeat": {"score": 0, "notes": "string"},
    "freshness": {"score": 0, "notes": "string"},
    "content_depth": {"score": 0, "notes": "string"},
    "scannability": {"score": 0, "notes": "string"},
    "ai_search_readiness": {"score": 0, "notes": "string"},
    "strengths": ["string"],
    "weaknesses": ["string"],
    "opportunities": ["string"],
    "overall_score": 0
  },
  "keyword_analysis": [
    {
      "keyword": "string",
      "coverage": "strong|partial|missing",
      "mentions": 0,
      "importance": "primary|secondary|supporting",
      "intent": "Informational|Commercial Investigation|Transactional|Navigational",
      "recommendation": "string"
    }
  ],
  "content_gap_analysis": {
    "gaps": [
      {
        "gap": "string",
        "why_it_matters": "string",
        "priority": "High|Medium|Low",
        "type": "statistics|case_study|faq|images|schema|examples|impact|internal_links|external_sources|depth|other"
      }
    ],
    "do_not_add": ["generic sections that would NOT fit this article"]
  },
  "heading_optimization": [
    {
      "current": "string",
      "suggested": "string or empty if keep",
      "reason": "string",
      "apply": true
    }
  ],
  "faqs": [
    {"question": "string", "answer": "string grounded in the article only"}
  ],
  "featured_snippets": {
    "suitable": true,
    "types": ["Definition|List|Table|Step|Comparison"],
    "definition": "string or empty",
    "list": ["string"],
    "table_note": "string or empty",
    "step": ["string"],
    "comparison_note": "string or empty"
  },
  "metadata": {
    "seo_title": "string <= 60 chars preferred",
    "meta_description": "string 120-160 chars",
    "slug": "kebab-case",
    "og_title": "string",
    "og_description": "string",
    "twitter_title": "string",
    "twitter_description": "string"
  },
  "links": {
    "internal": [{"anchor_text": "string", "target_topic": "string", "reason": "string"}],
    "external": [{"anchor_text": "string", "target_type": "string", "reason": "string", "example_domain_hint": "string"}]
  },
  "ai_search_optimization": {
    "google_ai_overview": ["string"],
    "perplexity": ["string"],
    "chatgpt_search": ["string"],
    "gemini": ["string"],
    "copilot": ["string"],
    "summary_improvements": ["string"],
    "direct_answers": ["string"],
    "entity_rich_paragraphs": ["string"],
    "question_headings": ["string"],
    "citation_opportunities": ["string"]
  },
  "optimization_plan": {
    "high_priority": [{"action": "string", "why": "string", "seo_impact": "string", "difficulty": "Easy|Medium|Hard"}],
    "medium_priority": [{"action": "string", "why": "string", "seo_impact": "string", "difficulty": "Easy|Medium|Hard"}],
    "low_priority": [{"action": "string", "why": "string", "seo_impact": "string", "difficulty": "Easy|Medium|Hard"}]
  },
  "quality_validation": {
    "topic_preserved": true,
    "search_intent_preserved": true,
    "no_hallucinated_keywords": true,
    "no_generic_sections": true,
    "metadata_matches_article": true,
    "faqs_relevant": true,
    "internal_links_make_sense": true,
    "heading_suggestions_relevant": true,
    "no_fabricated_content": true,
    "notes": ["string"],
    "passed": true
  },
  "final_seo_score": 0,
  "suggestions": ["short actionable bullets"]
}
"""


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _extract_json(raw: str) -> dict[str, Any]:
  text = (raw or "").strip()
  if not text:
    raise ValueError("empty optimizer strategist response")
  fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
  if fence:
    text = fence.group(1).strip()
  start = text.find("{")
  end = text.rfind("}")
  if start < 0 or end <= start:
    raise ValueError("no JSON object in optimizer strategist response")
  return json.loads(text[start : end + 1])


def _slugify(text: str) -> str:
  s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
  return s[:80] or "article"


def _as_list(val: Any) -> list[Any]:
  if isinstance(val, list):
    return val
  if val is None:
    return []
  return [val]


def _as_str(val: Any, default: str = "") -> str:
  if val is None:
    return default
  return str(val).strip() or default


def _clip(text: str, n: int) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  if len(t) <= n:
    return t
  return t[: n - 3].rsplit(" ", 1)[0] + "..."


def build_optimizer_strategist_prompt(
  *,
  title: str,
  content: str,
  keywords: list[str],
  category: str,
  tone: str,
  language: str,
  local_topic: dict[str, Any] | None = None,
) -> str:
  local_hint = ""
  if local_topic:
    local_hint = (
      f"\nLocal topic-resolution hint (validate; do not blindly trust):\n"
      f"- candidate_primary: {local_topic.get('primary_keyword')}\n"
      f"- confidence: {local_topic.get('confidence')}\n"
      f"- source: {local_topic.get('source')}\n"
      f"- candidates: {', '.join((local_topic.get('candidates_considered') or [])[:6])}\n"
    )
  kw_line = ", ".join(keywords[:12]) if keywords else "(none — infer from article)"
  return (
    f"Language: {language or 'English'}\n"
    f"Requested category hint: {category or 'blog_article'}\n"
    f"Requested tone hint: {tone or 'professional'}\n"
    f"User keywords (optional): {kw_line}\n"
    f"Title / H1: {title or '(extract from content)'}\n"
    f"{local_hint}\n"
    f"ARTICLE:\n{(content or '')[:9000]}\n\n"
    "Analyze this article according to the system rules. "
    "Preserve purpose and intent. Return the JSON schema only."
  )


def _flatten_entities(entity_analysis: dict[str, Any]) -> list[str]:
  out: list[str] = []
  seen: set[str] = set()
  for key in (
    "companies", "people", "locations", "products", "organizations",
    "events", "technologies", "services", "concepts",
  ):
    for item in _as_list(entity_analysis.get(key)):
      e = _clean(str(item))
      if not e or len(e) < 2:
        continue
      # Reject sentence fragments
      if e.lower().endswith((" is", " are", " the", " a", " an", " of", " for", " to")):
        continue
      if e.count(" ") > 6:
        continue
      low = e.lower()
      if low in seen:
        continue
      seen.add(low)
      out.append(e)
  return out[:40]


def _normalize_faqs(raw: Any, content: str) -> list[dict[str, str]]:
  faqs: list[dict[str, str]] = []
  content_low = (content or "").lower()
  bad_q = re.compile(
    r"^(what is|how does|benefits of)\s+.+\??$",
    re.I,
  )
  for item in _as_list(raw):
    if not isinstance(item, dict):
      continue
    q = _clean(_as_str(item.get("question")))
    a = _clean(_as_str(item.get("answer")))
    if not q or not a or len(a) < 20:
      continue
    # Soft-reject ultra-generic patterns when answer has no content overlap
    if bad_q.match(q) and not any(w in content_low for w in q.lower().split() if len(w) > 4):
      continue
    faqs.append({"question": q[:180], "answer": _clip(a, 420)})
    if len(faqs) >= 8:
      break
  return faqs


def _normalize_links(raw: Any) -> list[dict[str, str]]:
  links: list[dict[str, str]] = []
  for item in _as_list(raw):
    if not isinstance(item, dict):
      continue
    anchor = _clean(_as_str(item.get("anchor_text")))
    target = _clean(_as_str(item.get("target_topic") or item.get("target_type")))
    reason = _clean(_as_str(item.get("reason")))
    if not anchor or not target:
      continue
    # Reject sentence fragments as anchors
    if len(anchor.split()) > 6 or anchor.lower().endswith((" is", " the", " a", " for")):
      continue
    links.append({
      "anchor_text": anchor[:80],
      "target_topic": target[:120],
      "reason": reason[:200] or "Relevant topical link.",
    })
    if len(links) >= 10:
      break
  return links


def _plan_to_suggestions(plan: dict[str, Any]) -> list[str]:
  out: list[str] = []
  for bucket in ("high_priority", "medium_priority", "low_priority"):
    for item in _as_list(plan.get(bucket)):
      if isinstance(item, dict) and item.get("action"):
        prefix = bucket.replace("_priority", "").title()
        out.append(f"[{prefix}] {_clean(str(item['action']))}")
  return out[:14]


def strategist_payload_to_result(
  data: dict[str, Any],
  *,
  content: str,
  keywords: list[str],
  category: str,
  tone: str,
  language: str,
  elapsed_ms: float,
  backend: str,
  local_topic: dict[str, Any] | None = None,
) -> dict[str, Any]:
  understanding = data.get("article_understanding") if isinstance(data.get("article_understanding"), dict) else {}
  entities_block = data.get("entity_analysis") if isinstance(data.get("entity_analysis"), dict) else {}
  topic_res = data.get("topic_resolution") if isinstance(data.get("topic_resolution"), dict) else {}
  seo_audit = data.get("seo_audit") if isinstance(data.get("seo_audit"), dict) else {}
  gaps_block = data.get("content_gap_analysis") if isinstance(data.get("content_gap_analysis"), dict) else {}
  metadata_raw = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
  links_raw = data.get("links") if isinstance(data.get("links"), dict) else {}
  ai_opt = data.get("ai_search_optimization") if isinstance(data.get("ai_search_optimization"), dict) else {}
  plan = data.get("optimization_plan") if isinstance(data.get("optimization_plan"), dict) else {}
  validation = data.get("quality_validation") if isinstance(data.get("quality_validation"), dict) else {}
  featured = data.get("featured_snippets") if isinstance(data.get("featured_snippets"), dict) else {}

  primary = _clean(
    _as_str(topic_res.get("primary_keyword"))
    or _as_str(topic_res.get("primary_seo_topic"))
    or _as_str(understanding.get("primary_topic"))
    or (local_topic or {}).get("primary_keyword")
    or (keywords[0] if keywords else "")
    or _extract_h1(content)
    or "article topic"
  )
  secondary = [
    _clean(str(x)) for x in _as_list(topic_res.get("secondary_keywords")) if _clean(str(x))
  ][:8]
  if primary.lower() in {s.lower() for s in secondary}:
    secondary = [s for s in secondary if s.lower() != primary.lower()]

  resolved_kws = [primary] + [s for s in secondary if s.lower() != primary.lower()]
  for k in keywords:
    if k and k.lower() not in {x.lower() for x in resolved_kws}:
      resolved_kws.append(k)
  resolved_kws = resolved_kws[:12]

  entities = _flatten_entities(entities_block)
  if not entities:
    entities = extract_entities_from_content(content, resolved_kws)[:20]

  faqs = _normalize_faqs(data.get("faqs"), content)
  internal_links = _normalize_links(links_raw.get("internal"))
  external_links = _normalize_links(links_raw.get("external"))

  title = _clip(
    _as_str(metadata_raw.get("seo_title")) or _extract_h1(content) or primary,
    70,
  )
  meta_desc = _clip(
    _as_str(metadata_raw.get("meta_description"))
    or _clip(re.split(r"[.\n]", content, maxsplit=1)[0], 155),
    160,
  )
  slug = _as_str(metadata_raw.get("slug")) or _slugify(title)
  metadata = {
    "title": title,
    "meta_description": meta_desc,
    "slug": slug,
    "og_title": _clip(_as_str(metadata_raw.get("og_title")) or title, 60),
    "og_description": _clip(_as_str(metadata_raw.get("og_description")) or meta_desc, 160),
    "twitter_title": _clip(_as_str(metadata_raw.get("twitter_title")) or title, 60),
    "twitter_description": _clip(_as_str(metadata_raw.get("twitter_description")) or meta_desc, 160),
  }

  kw_rows_raw = data.get("keyword_analysis")
  kw_table: list[dict[str, Any]] = []
  for row in _as_list(kw_rows_raw):
    if not isinstance(row, dict):
      continue
    kw = _clean(_as_str(row.get("keyword")))
    if not kw:
      continue
    mentions = row.get("mentions")
    if not isinstance(mentions, int):
      mentions = content.lower().count(kw.lower())
    kw_table.append({
      "keyword": kw,
      "status": _as_str(row.get("coverage") or row.get("importance") or "secondary"),
      "mentions": mentions,
      "importance": _as_str(row.get("importance")),
      "intent": _as_str(row.get("intent")),
      "recommendation": _as_str(row.get("recommendation")),
      "sections": "",
    })

  gap_rows = []
  for g in _as_list(gaps_block.get("gaps")):
    if not isinstance(g, dict):
      continue
    suggestion = _as_str(g.get("gap") or g.get("suggestion"))
    if not suggestion:
      continue
    gap_rows.append({
      "type": _as_str(g.get("type") or "other"),
      "priority": _as_str(g.get("priority") or "Medium"),
      "suggestion": suggestion,
      "why": _as_str(g.get("why_it_matters")),
    })

  strengths = [_clean(str(x)) for x in _as_list(seo_audit.get("strengths")) if _clean(str(x))]
  weaknesses = [_clean(str(x)) for x in _as_list(seo_audit.get("weaknesses")) if _clean(str(x))]
  opportunities = [_clean(str(x)) for x in _as_list(seo_audit.get("opportunities")) if _clean(str(x))]

  original_metrics = content_metrics(content)
  issues_before = analyze_issues(content, resolved_kws)
  seo_before = seo_score_from_analysis(original_metrics, issues_before)

  try:
    final_score = int(data.get("final_seo_score") or seo_audit.get("overall_score") or seo_before)
  except (TypeError, ValueError):
    final_score = seo_before
  final_score = max(0, min(100, final_score))

  # Audit/plan mode: do not rewrite body
  optimized = content
  optimized_metrics = original_metrics
  issues_after = issues_before
  seo_after = final_score

  ai_search_lines: list[str] = []
  for key in (
    "summary_improvements", "direct_answers", "entity_rich_paragraphs",
    "question_headings", "citation_opportunities",
    "google_ai_overview", "perplexity", "chatgpt_search", "gemini", "copilot",
  ):
    for item in _as_list(ai_opt.get(key)):
      line = _clean(str(item))
      if line and line not in ai_search_lines:
        ai_search_lines.append(line)

  list_snip = featured.get("list")
  if isinstance(list_snip, list):
    list_text = "\n".join(f"- {x}" for x in list_snip[:8] if str(x).strip())
  else:
    list_text = _as_str(list_snip)

  suggestions = [_clean(str(x)) for x in _as_list(data.get("suggestions")) if _clean(str(x))]
  suggestions = (suggestions + _plan_to_suggestions(plan))[:14]

  conf = topic_res.get("confidence")
  try:
    conf_i = int(conf) if conf is not None else int(understanding.get("confidence_score") or 0)
  except (TypeError, ValueError):
    conf_i = 0

  topic_resolution_stage = {
    "primary_keyword": primary,
    "display_title": _as_str(topic_res.get("primary_seo_topic")) or primary,
    "confidence": conf_i,
    "source": "hosted_strategist",
    "secondary_keywords": secondary,
    "semantic_topics": [_clean(str(x)) for x in _as_list(topic_res.get("semantic_topics"))[:10]],
    "long_tail_topics": [_clean(str(x)) for x in _as_list(topic_res.get("long_tail_topics"))[:8]],
    "question_topics": [_clean(str(x)) for x in _as_list(topic_res.get("question_topics"))[:8]],
    "content_clusters": [_clean(str(x)) for x in _as_list(topic_res.get("content_clusters"))[:6]],
    "local_hint": (local_topic or {}).get("primary_keyword"),
  }

  seo_report = {
    "article_understanding": understanding,
    "seo_audit": {
      "strengths": strengths,
      "weaknesses": weaknesses,
      "opportunities": opportunities,
      "scores": {k: v for k, v in seo_audit.items() if k not in ("strengths", "weaknesses", "opportunities")},
      "overall_score": final_score,
    },
    "keyword_analysis": {
      "primary": primary,
      "secondary": secondary,
      "table": kw_table,
    },
    "entity_analysis": {
      "entities": [{"entity": e, "in_content": e.lower() in content.lower()} for e in entities[:16]],
      "entity_coverage_pct": round(
        100 * sum(1 for e in entities[:12] if e.lower() in content.lower()) / max(1, min(len(entities), 12)),
        1,
      ),
      "grouped": entities_block,
    },
    "content_gap_analysis": {
      "gaps": gap_rows,
      "do_not_add": [_clean(str(x)) for x in _as_list(gaps_block.get("do_not_add"))[:8]],
    },
    "heading_optimization": [
      x for x in _as_list(data.get("heading_optimization")) if isinstance(x, dict)
    ][:12],
    "search_intent": {
      "primary": _as_str(understanding.get("search_intent")) or "Informational",
      "article_type": _as_str(understanding.get("article_type")),
      "audience": _as_str(understanding.get("target_audience")),
    },
    "featured_snippets": {
      "suitable": bool(featured.get("suitable", True)),
      "definition": _as_str(featured.get("definition")),
      "list": list_text,
      "table": _as_str(featured.get("table_note")),
      "step": _as_list(featured.get("step")),
      "faq": faqs[:3],
    },
    "faqs": faqs,
    "internal_links": internal_links,
    "external_links": external_links,
    "metadata": metadata,
    "optimization_plan": plan,
    "quality_validation": validation,
    "ai_search_optimization": ai_search_lines[:12],
    "final_metrics": {
      "seo_score_before": seo_before,
      "seo_score_after": seo_after,
      "readability_score": original_metrics.get("readability_score"),
      "keyword_coverage_pct": None,
      "entity_coverage_pct": round(
        100 * sum(1 for e in entities[:12] if e.lower() in content.lower()) / max(1, min(len(entities), 12)),
        1,
      ),
      "content_depth_score": min(100, round(original_metrics.get("word_count", 0) / 8)),
      "topical_authority_score": conf_i,
      "rewrite_applied": False,
    },
  }

  return {
    "category": category,
    "language": language or "en",
    "tone": tone,
    "original": original_metrics,
    "optimized": optimized_metrics,
    "metrics": {"original": original_metrics, "optimized": optimized_metrics},
    "seo_score_before": seo_before,
    "seo_score_after": seo_after,
    "improvement": seo_after - seo_before,
    "optimized_content": optimized,
    "suggestions": suggestions,
    "issues_before": issues_before,
    "issues_after": issues_after,
    "keywords": resolved_kws,
    "ai": {
      "enabled": True,
      "model_used": True,
      "backend": backend,
      "hosted": True,
      "mode": "optimizer_strategist_v1",
    },
    "use_rag": False,
    "generator_version": STRATEGIST_VERSION,
    "variation_seed": None,
    "rewrite_applied": False,
    "article_understanding": understanding,
    "optimization_plan": plan,
    "quality_validation": validation,
    "seo_report": seo_report,
    "architecture": {
      "flow": [
        "input_article",
        "article_understanding",
        "entity_analysis",
        "topic_resolution",
        "seo_audit",
        "keyword_analysis",
        "content_gap_analysis",
        "heading_optimization",
        "faq_generation",
        "featured_snippets",
        "metadata",
        "link_optimization",
        "ai_search_optimization",
        "optimization_plan",
        "quality_validation",
      ],
      "stages": {
        "topic_resolution": topic_resolution_stage,
        "article_understanding": understanding,
        "entity_extraction": {"from_strategist": entities, "grouped": entities_block},
        "seo_audit": seo_audit,
        "quality_validation": validation,
      },
    },
    "pipeline": {
      "keyword_analysis": {"primary": primary, "secondary": secondary, "table": kw_table},
      "entity_extraction": entities,
      "coverage_map": {},
      "gap_analysis": gap_rows,
      "source_router": {"mode": "hosted_strategist"},
      "retrieval": {"sources_used": [], "document_count": 0, "confidence": conf_i / 100.0},
      "novelty": {},
      "section_plan": [],
      "readability_analysis": original_metrics,
    },
    "optimization": {
      "metadata": {
        "title": metadata["title"],
        "meta_description": metadata["meta_description"],
      },
      "internal_links": internal_links,
      "faqs": faqs,
      "schema_suggestions": {},
      "seo_report": seo_report,
      "external_links": external_links,
      "heading_optimization": seo_report["heading_optimization"],
      "featured_snippets": seo_report["featured_snippets"],
    },
    "rag": {"enabled": False, "sources_used": [], "confidence": conf_i / 100.0},
    "elapsed_ms": round(elapsed_ms, 1),
    "strategist": data,
  }


async def generate_with_optimizer_strategist(
  provider: ModelProvider,
  *,
  content: str,
  keywords: list[str] | None = None,
  category: str = "blog_article",
  tone: str = "professional",
  language: str = "English",
) -> dict[str, Any]:
  content = (content or "").strip()
  if not content:
    raise ValueError("content is required")
  kws = [k for k in (keywords or []) if k and str(k).strip()][:20]
  title = _extract_h1(content) or ""
  local_topic = resolve_seo_topic(content, keywords=kws, display_title=title)

  t0 = time.perf_counter()
  user_prompt = build_optimizer_strategist_prompt(
    title=title,
    content=content,
    keywords=kws,
    category=category,
    tone=tone,
    language=language,
    local_topic=local_topic,
  )
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=_SYSTEM,
    use_rag=False,
    skip_intent=True,
    skip_kb_direct_match=True,
    max_tokens=5500,
    temperature=0.3,
  )
  data = _extract_json(raw)
  backend = getattr(provider, "last_backend", None) or getattr(provider, "model_id", lambda: "hosted")()
  if callable(backend):
    try:
      backend = backend()
    except TypeError:
      pass
  elapsed = (time.perf_counter() - t0) * 1000
  return strategist_payload_to_result(
    data,
    content=content,
    keywords=kws,
    category=category,
    tone=tone,
    language=language or "en",
    elapsed_ms=elapsed,
    backend=str(backend) if backend else "hosted",
    local_topic=local_topic,
  )
