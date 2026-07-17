"""Elite SEO Content Optimizer Strategist v2 — agency consultant mode.

Hosted LLM: understand → audit → plan → conditional (minimal) rewrite.
Never regenerates an article into a generic tutorial.
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

STRATEGIST_VERSION = "seo-optimizer-strategist-v2"

_SYSTEM = """You are an Elite SEO Content Optimizer used by professional SEO agencies.

Your job is to analyze and optimize existing content for Google Search, Google AI Overviews, Bing Copilot, ChatGPT Search, Perplexity, Gemini and other semantic search engines.

You are NOT a blog writer.
You are NOT a generic content generator.
You are an SEO optimization strategist.

## PRIMARY OBJECTIVE
Improve SEO quality of an EXISTING article while preserving:
• Original topic, purpose, author intent, audience, writing style, factual content

Optimization should feel like an editor improving the content, not an AI rewriting it.
Never transform an article into a completely different article.

## STRICT RULES
1. Never change the article topic.
   Wrong: rewrite "Coffey Bros Moving's community involvement" into "Corporate Social Responsibility Guide".
2. Never invent generic sections unless already relevant / article type supports them:
   Forbidden by default: Best Practices, Features, How To, Pros & Cons, Checklist, Comparison, Advantages, Disadvantages.
3. Never force keywords. Keywords must naturally support the article.
4. Never hallucinate facts. If missing, recommend adding — do NOT fabricate.
5. Never use first sentence / first noun phrase / opening paragraph alone as the topic — use semantic understanding.
6. Never output meaningless keywords or sentence fragments.
   Wrong: "Success Business Often", "For Coffey Bros Moving", "Throughout the Chicago"
   Correct: "Community involvement", "Chicago moving company", "Corporate social responsibility"

## ARTICLE TYPE CONTROLS OPTIMIZATION
Classify exactly one:
Blog | News | Case Study | Brand Story | Press Release | Opinion | Educational Guide | Tutorial | Landing Page | Product Page | Research Article | Documentation | Review | Comparison

Examples:
• Tutorial → may recommend How To
• Product → may recommend Features
• Brand Story → NEVER recommend How To / Best Practices / Features
• Case Study → NEVER recommend Features
• News → NEVER recommend Best Practices

## FAQ RULES
Never generate generic FAQs.
Wrong: "What is Coffey Bros Moving?", "How does Coffey Bros Moving work?", "Benefits of Coffey Bros Moving?"
Correct: "How does Coffey Bros Moving support Chicago communities?", "Why is community involvement important for businesses?"

## METADATA RULES
Must accurately summarize THIS article. Never append Tips / Strategies / Guide / Best Practices unless appropriate for article type.

## CONDITIONAL REWRITE (CRITICAL)
Do NOT rewrite the whole article.
Only rewrite if necessary. Allowed: title/H1, meta, headings, introduction, conclusion, weak paragraphs, transitions, natural keyword placement.
If already high quality, set rewrite_required=false and summary="No rewrite required."
If rewriting, provide minimal patches and optionally optimized_markdown that PRESERVES structure and facts — never invent new How To / Features / Best Practices sections.
optimized_markdown must remain recognizably the same article (same entities, same story, same purpose).

## MOST IMPORTANT
Behave like an experienced SEO consultant reviewing an existing article.
Every recommendation must be specific to THIS article.
If a recommendation could apply to any article on the internet, reject it.

Return JSON only. No markdown fences. No commentary.

Schema:
{
  "article_understanding": {
    "primary_topic": "string",
    "secondary_topics": ["string"],
    "article_type": "Blog|News|Case Study|Brand Story|Press Release|Opinion|Educational Guide|Tutorial|Landing Page|Product Page|Research Article|Documentation|Review|Comparison",
    "search_intent": "Informational|Commercial Investigation|Transactional|Navigational",
    "industry": "string from content only",
    "target_audience": "string",
    "tone": "string",
    "content_goal": "string",
    "customer_journey_stage": "Awareness|Consideration|Decision|Retention|Advocacy",
    "confidence_score": 0
  },
  "topic_resolution": {
    "primary_seo_topic": "string",
    "primary_keyword": "string",
    "supporting_keywords": ["up to 8"],
    "semantic_topics": ["up to 10"],
    "entities": ["named entities"],
    "confidence": 0
  },
  "seo_audit": {
    "title": {"score": 0, "notes": "string"},
    "meta_title": {"score": 0, "notes": "string"},
    "meta_description": {"score": 0, "notes": "string"},
    "slug": {"score": 0, "notes": "string"},
    "headings": {"score": 0, "notes": "string"},
    "keyword_usage": {"score": 0, "notes": "string"},
    "entity_coverage": {"score": 0, "notes": "string"},
    "semantic_coverage": {"score": 0, "notes": "string"},
    "readability": {"score": 0, "notes": "string"},
    "eeat": {"score": 0, "notes": "string"},
    "content_freshness": {"score": 0, "notes": "string"},
    "content_depth": {"score": 0, "notes": "string"},
    "internal_linking": {"score": 0, "notes": "string"},
    "external_linking": {"score": 0, "notes": "string"},
    "images": {"score": 0, "notes": "string"},
    "ai_search_readiness": {"score": 0, "notes": "string"},
    "strengths": ["article-specific"],
    "weaknesses": ["article-specific"],
    "opportunities": ["article-specific"],
    "overall_score": 0
  },
  "keyword_analysis": [
    {
      "keyword": "string",
      "coverage": "strong|partial|missing",
      "mentions": 0,
      "intent": "Informational|Commercial Investigation|Transactional|Navigational",
      "importance": "primary|secondary|supporting",
      "recommendation": "string"
    }
  ],
  "entity_analysis": {
    "people": [],
    "companies": [],
    "locations": [],
    "organizations": [],
    "products": [],
    "services": [],
    "technologies": [],
    "concepts": [],
    "events": [],
    "covered": [],
    "missing": [],
    "overused": []
  },
  "content_gap_analysis": {
    "gaps": [
      {
        "gap": "string",
        "why_it_matters": "string",
        "priority": "High|Medium|Low",
        "type": "statistics|case_study|examples|faq|internal_links|schema|images|outcomes|depth|other"
      }
    ],
    "do_not_add": ["generic sections that do NOT fit this article type"]
  },
  "heading_analysis": [
    {
      "current": "string",
      "action": "Keep|Improve|Merge|Split|Remove",
      "suggested": "string or empty",
      "reason": "string"
    }
  ],
  "faqs": [
    {"question": "string", "answer": "string grounded only in the article"}
  ],
  "featured_snippets": {
    "suitable": true,
    "types": ["Definition|List|Table|Step|Comparison"],
    "definition": "string or empty",
    "list": ["string"],
    "table_note": "string or empty",
    "step": ["string"]
  },
  "metadata": {
    "seo_title": "string",
    "meta_description": "string 120-160 chars",
    "slug": "kebab-case",
    "og_title": "string",
    "og_description": "string",
    "twitter_title": "string",
    "twitter_description": "string"
  },
  "link_optimization": {
    "internal": [{"anchor_text": "string", "target_topic": "string", "reason": "string"}],
    "external": [{"anchor_text": "string", "target_type": "string", "reason": "string", "example_domain_hint": "string"}],
    "image_alt_text": [{"suggestion": "string", "reason": "string"}]
  },
  "schema_recommendations": [
    {"schema_type": "string", "why": "string", "priority": "High|Medium|Low"}
  ],
  "ai_search_optimization": {
    "google_ai_overview": ["string"],
    "perplexity": ["string"],
    "gemini": ["string"],
    "chatgpt_search": ["string"],
    "copilot": ["string"],
    "answer_first_paragraphs": ["string"],
    "entity_rich_summaries": ["string"],
    "citation_opportunities": ["string"],
    "question_headings": ["string"]
  },
  "optimization_plan": {
    "high_priority": [{"problem": "string", "recommendation": "string", "seo_benefit": "string", "difficulty": "Easy|Medium|Hard", "estimated_impact": "Low|Medium|High"}],
    "medium_priority": [{"problem": "string", "recommendation": "string", "seo_benefit": "string", "difficulty": "Easy|Medium|Hard", "estimated_impact": "Low|Medium|High"}],
    "low_priority": [{"problem": "string", "recommendation": "string", "seo_benefit": "string", "difficulty": "Easy|Medium|Hard", "estimated_impact": "Low|Medium|High"}]
  },
  "conditional_rewrite": {
    "rewrite_required": true,
    "summary": "No rewrite required. | Minimal editorial improvements applied.",
    "title_h1": "string or empty",
    "introduction": "string or empty — improved intro only if needed",
    "conclusion": "string or empty",
    "heading_replacements": [{"from": "string", "to": "string"}],
    "paragraph_replacements": [{"from": "string", "to": "string"}],
    "optimized_markdown": "optional lightly edited full article OR empty — must preserve topic/facts/structure; no invented generic sections"
  },
  "quality_validation": {
    "topic_preserved": true,
    "search_intent_preserved": true,
    "no_hallucinated_facts": true,
    "no_generic_sections": true,
    "no_forced_keywords": true,
    "metadata_matches_article": true,
    "faqs_match_article": true,
    "heading_suggestions_fit": true,
    "rewrite_is_minimal": true,
    "notes": ["string"],
    "passed": true
  },
  "final_seo_score": 0,
  "suggestions": ["short article-specific bullets"]
}
"""

_GENERIC_SECTION_RE = re.compile(
  r"(?i)^\s{0,3}#{1,3}\s+(best practices|features|how to(?:\s|$)|pros\s*(?:&|and)\s*cons|"
  r"checklist|comparison|advantages|disadvantages)\b"
)


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
  apply_conditional_rewrite: bool = True,
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
  rewrite_note = (
    "Provide conditional_rewrite patches and, if helpful, a lightly edited optimized_markdown "
    "that preserves the original article. Do NOT regenerate into a tutorial."
    if apply_conditional_rewrite
    else "Set conditional_rewrite.rewrite_required=false and leave optimized_markdown empty."
  )
  kw_line = ", ".join(keywords[:12]) if keywords else "(none — infer from article)"
  return (
    f"Language: {language or 'English'}\n"
    f"Requested category hint: {category or 'blog_article'}\n"
    f"Requested tone hint: {tone or 'professional'}\n"
    f"User keywords (optional): {kw_line}\n"
    f"Title / H1: {title or '(extract from content)'}\n"
    f"{local_hint}\n"
    f"Rewrite instruction: {rewrite_note}\n\n"
    f"ARTICLE:\n{(content or '')[:9000]}\n\n"
    "Act as an SEO consultant for THIS article only. Return the JSON schema only."
  )


def _flatten_entities(entity_analysis: dict[str, Any]) -> list[str]:
  out: list[str] = []
  seen: set[str] = set()
  keys = (
    "covered", "companies", "people", "locations", "products", "organizations",
    "events", "technologies", "services", "concepts",
  )
  for key in keys:
    for item in _as_list(entity_analysis.get(key)):
      e = _clean(str(item))
      if not e or len(e) < 2:
        continue
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
  bad_q = re.compile(r"^(what is|how does|benefits of)\s+.+\??$", re.I)
  for item in _as_list(raw):
    if not isinstance(item, dict):
      continue
    q = _clean(_as_str(item.get("question")))
    a = _clean(_as_str(item.get("answer")))
    if not q or not a or len(a) < 20:
      continue
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
      if not isinstance(item, dict):
        continue
      action = _clean(_as_str(item.get("recommendation") or item.get("action")))
      if not action:
        continue
      prefix = bucket.replace("_priority", "").title()
      out.append(f"[{prefix}] {action}")
  return out[:14]


def _strip_invented_generic_sections(markdown: str, article_type: str) -> str:
  """Drop newly invented forbidden H2/H3 blocks for non-tutorial/product types."""
  at = (article_type or "").lower()
  allow = any(x in at for x in ("tutorial", "product", "comparison", "educational guide", "landing"))
  if allow:
    return markdown
  lines = (markdown or "").splitlines()
  out: list[str] = []
  skipping = False
  for line in lines:
    if _GENERIC_SECTION_RE.match(line):
      skipping = True
      continue
    if skipping and re.match(r"^\s{0,3}#{1,3}\s+\S", line):
      skipping = False
    if skipping and re.match(r"^\s{0,3}#{1,3}\s+", line):
      skipping = False
    if skipping:
      # stop skip at next heading of same/higher level already handled; continue skip for body
      if re.match(r"^\s{0,3}#{1,6}\s+", line):
        skipping = False
        out.append(line)
      continue
    out.append(line)
  return "\n".join(out).strip()


def apply_conditional_rewrite(
  content: str,
  conditional: dict[str, Any] | None,
  *,
  article_type: str = "",
  metadata_title: str = "",
  faqs: list[dict[str, str]] | None = None,
) -> tuple[str, dict[str, Any]]:
  """Apply minimal editorial patches. Never invent a new article."""
  info: dict[str, Any] = {
    "rewrite_required": False,
    "summary": "No rewrite required.",
    "patches_applied": [],
  }
  if not isinstance(conditional, dict):
    return content, info

  required = bool(conditional.get("rewrite_required"))
  summary = _as_str(conditional.get("summary")) or (
    "Minimal editorial improvements applied." if required else "No rewrite required."
  )
  info["rewrite_required"] = required
  info["summary"] = summary

  if not required and not _as_str(conditional.get("optimized_markdown")):
    # Still allow H1 sync from metadata title if present and content has H1
    text = content
    h1 = _as_str(conditional.get("title_h1")) or metadata_title
    if h1 and re.search(r"^#\s+.+$", text, re.M):
      new_text, n = re.subn(r"^#\s+.+$", f"# {h1}", text, count=1, flags=re.M)
      if n:
        text = new_text
        info["patches_applied"].append("title_h1")
        info["rewrite_required"] = True
        info["summary"] = "Minimal title alignment applied."
    return text, info

  # Prefer full lightly-edited markdown when it looks like the same article
  opt_md = _as_str(conditional.get("optimized_markdown"))
  if opt_md and len(opt_md) >= max(80, int(len(content) * 0.45)):
    # Reject if it balloons into a different longform piece
    if len(opt_md) <= int(len(content) * 2.2) + 800:
      cleaned = _strip_invented_generic_sections(opt_md, article_type)
      info["patches_applied"].append("optimized_markdown")
      info["rewrite_required"] = True
      # Optionally append FAQs if not already present
      text = cleaned
      if faqs and not re.search(r"(?i)^##\s+frequently asked questions", text, re.M):
        faq_block = ["", "## Frequently Asked Questions", ""]
        for f in faqs[:5]:
          faq_block.append(f"**{f['question']}**")
          faq_block.append(f["answer"])
          faq_block.append("")
        text = text.rstrip() + "\n" + "\n".join(faq_block).rstrip()
        info["patches_applied"].append("faqs_appended")
      return text.strip(), info

  text = content

  h1 = _as_str(conditional.get("title_h1")) or metadata_title
  if h1:
    if re.search(r"^#\s+.+$", text, re.M):
      text, n = re.subn(r"^#\s+.+$", f"# {h1}", text, count=1, flags=re.M)
      if n:
        info["patches_applied"].append("title_h1")
    else:
      text = f"# {h1}\n\n{text}"
      info["patches_applied"].append("title_h1_inserted")

  for item in _as_list(conditional.get("heading_replacements")):
    if not isinstance(item, dict):
      continue
    frm = _as_str(item.get("from"))
    to = _as_str(item.get("to"))
    if not frm or not to or frm == to:
      continue
    pattern = re.compile(rf"^(#{{1,6}})\s*{re.escape(frm)}\s*$", re.M)
    text2, n = pattern.subn(rf"\1 {to}", text, count=1)
    if n:
      text = text2
      info["patches_applied"].append(f"heading:{frm[:40]}")

  for item in _as_list(conditional.get("paragraph_replacements")):
    if not isinstance(item, dict):
      continue
    frm = _as_str(item.get("from"))
    to = _as_str(item.get("to"))
    if not frm or not to or len(frm) < 20:
      continue
    if frm in text:
      text = text.replace(frm, to, 1)
      info["patches_applied"].append("paragraph")

  intro = _as_str(conditional.get("introduction"))
  if intro and len(intro) > 40:
    # Replace first non-heading paragraph block after H1
    parts = text.split("\n")
    i = 0
    while i < len(parts) and (not parts[i].strip() or parts[i].lstrip().startswith("#")):
      i += 1
    if i < len(parts):
      # find end of first paragraph
      j = i
      while j < len(parts) and parts[j].strip() and not parts[j].lstrip().startswith("#"):
        j += 1
      parts = parts[:i] + [intro] + parts[j:]
      text = "\n".join(parts)
      info["patches_applied"].append("introduction")

  conclusion = _as_str(conditional.get("conclusion"))
  if conclusion and len(conclusion) > 40:
    # Replace last non-heading paragraph, or append
    parts = [p for p in text.split("\n\n") if p.strip()]
    if parts:
      last = parts[-1]
      if not last.lstrip().startswith("#") and not last.lstrip().startswith("**"):
        parts[-1] = conclusion
        info["patches_applied"].append("conclusion")
      else:
        parts.append(conclusion)
        info["patches_applied"].append("conclusion_appended")
      text = "\n\n".join(parts)

  if faqs and not re.search(r"(?i)^##\s+frequently asked questions", text, re.M):
    # Only append FAQs when some rewrite happened or rewrite was requested
    if info["patches_applied"] or required:
      faq_block = ["## Frequently Asked Questions"]
      for f in faqs[:5]:
        faq_block.append(f"**{f['question']}**\n\n{f['answer']}")
      text = text.rstrip() + "\n\n" + "\n\n".join(faq_block)
      info["patches_applied"].append("faqs_appended")

  if info["patches_applied"]:
    info["rewrite_required"] = True
    if summary.lower().startswith("no rewrite"):
      info["summary"] = "Minimal editorial improvements applied."

  return text.strip(), info


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
  apply_rewrite: bool = True,
) -> dict[str, Any]:
  understanding = data.get("article_understanding") if isinstance(data.get("article_understanding"), dict) else {}
  entities_block = data.get("entity_analysis") if isinstance(data.get("entity_analysis"), dict) else {}
  topic_res = data.get("topic_resolution") if isinstance(data.get("topic_resolution"), dict) else {}
  seo_audit = data.get("seo_audit") if isinstance(data.get("seo_audit"), dict) else {}
  gaps_block = data.get("content_gap_analysis") if isinstance(data.get("content_gap_analysis"), dict) else {}
  metadata_raw = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
  links_raw = data.get("link_optimization") if isinstance(data.get("link_optimization"), dict) else {}
  if not links_raw and isinstance(data.get("links"), dict):
    links_raw = data["links"]
  ai_opt = data.get("ai_search_optimization") if isinstance(data.get("ai_search_optimization"), dict) else {}
  plan = data.get("optimization_plan") if isinstance(data.get("optimization_plan"), dict) else {}
  validation = data.get("quality_validation") if isinstance(data.get("quality_validation"), dict) else {}
  featured = data.get("featured_snippets") if isinstance(data.get("featured_snippets"), dict) else {}
  conditional = data.get("conditional_rewrite") if isinstance(data.get("conditional_rewrite"), dict) else {}
  schema_recs = [x for x in _as_list(data.get("schema_recommendations")) if isinstance(x, dict)]
  heading_analysis = [x for x in _as_list(data.get("heading_analysis") or data.get("heading_optimization")) if isinstance(x, dict)]

  article_type = _as_str(understanding.get("article_type")) or "Blog"

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
    _clean(str(x))
    for x in _as_list(topic_res.get("supporting_keywords") or topic_res.get("secondary_keywords"))
    if _clean(str(x))
  ][:8]
  secondary = [s for s in secondary if s.lower() != primary.lower()]

  resolved_kws = [primary] + secondary
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
  image_alts = [
    {"suggestion": _as_str(x.get("suggestion")), "reason": _as_str(x.get("reason"))}
    for x in _as_list(links_raw.get("image_alt_text"))
    if isinstance(x, dict) and _as_str(x.get("suggestion"))
  ][:8]

  title = _clip(
    _as_str(metadata_raw.get("seo_title")) or _extract_h1(content) or primary,
    70,
  )
  # Reject forced Guide/Tips suffixes for brand stories etc.
  if article_type.lower() in {"brand story", "news", "press release", "case study", "opinion"}:
    title = re.sub(
      r"(?i)(:|\||—|-)\s*(tips|strategies|guide|best practices).*$",
      "",
      title,
    ).strip(" :-|—") or title

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

  kw_table: list[dict[str, Any]] = []
  for row in _as_list(data.get("keyword_analysis")):
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
    # Filter forbidden generic gaps for brand/news types
    low = suggestion.lower()
    if article_type.lower() in {"brand story", "news", "press release", "case study", "opinion"}:
      if any(x in low for x in ("best practices", "how to section", "add features", "pros & cons", "pros and cons")):
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

  rewrite_info: dict[str, Any] = {
    "rewrite_required": False,
    "summary": "No rewrite required.",
    "patches_applied": [],
  }
  if apply_rewrite:
    optimized, rewrite_info = apply_conditional_rewrite(
      content,
      conditional,
      article_type=article_type,
      metadata_title=title,
      faqs=faqs,
    )
  else:
    optimized = content

  optimized_metrics = content_metrics(optimized)
  issues_after = analyze_issues(optimized, resolved_kws)
  seo_after = max(final_score, seo_score_from_analysis(optimized_metrics, issues_after))

  ai_search_lines: list[str] = []
  for key in (
    "answer_first_paragraphs", "entity_rich_summaries", "citation_opportunities",
    "question_headings", "google_ai_overview", "perplexity", "gemini",
    "chatgpt_search", "copilot",
    "summary_improvements", "direct_answers", "entity_rich_paragraphs",
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
  if rewrite_info.get("summary"):
    suggestions = [f"Rewrite: {rewrite_info['summary']}"] + [s for s in suggestions if s][:13]

  conf = topic_res.get("confidence")
  try:
    conf_i = int(conf) if conf is not None else int(understanding.get("confidence_score") or 0)
  except (TypeError, ValueError):
    conf_i = 0

  # Normalize journey stage field name for hub
  if understanding and not understanding.get("reader_journey_stage"):
    understanding["reader_journey_stage"] = understanding.get("customer_journey_stage")

  topic_resolution_stage = {
    "primary_keyword": primary,
    "display_title": _as_str(topic_res.get("primary_seo_topic")) or primary,
    "confidence": conf_i,
    "source": "hosted_strategist_v2",
    "secondary_keywords": secondary,
    "semantic_topics": [_clean(str(x)) for x in _as_list(topic_res.get("semantic_topics"))[:10]],
    "entities": [_clean(str(x)) for x in _as_list(topic_res.get("entities"))[:16]],
    "local_hint": (local_topic or {}).get("primary_keyword"),
  }

  covered = [_clean(str(x)) for x in _as_list(entities_block.get("covered")) if _clean(str(x))]
  missing = [_clean(str(x)) for x in _as_list(entities_block.get("missing")) if _clean(str(x))]
  overused = [_clean(str(x)) for x in _as_list(entities_block.get("overused")) if _clean(str(x))]
  if not covered:
    covered = [e for e in entities if e.lower() in content.lower()][:12]

  seo_report = {
    "article_understanding": understanding,
    "article_type": article_type,
    "seo_audit": {
      "strengths": strengths,
      "weaknesses": weaknesses,
      "opportunities": opportunities,
      "scores": {
        k: v for k, v in seo_audit.items()
        if k not in ("strengths", "weaknesses", "opportunities")
      },
      "overall_score": final_score,
    },
    "keyword_analysis": {
      "primary": primary,
      "secondary": secondary,
      "table": kw_table,
    },
    "entity_analysis": {
      "entities": [{"entity": e, "in_content": e.lower() in content.lower()} for e in entities[:16]],
      "covered": covered[:16],
      "missing": missing[:12],
      "overused": overused[:8],
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
    "heading_analysis": heading_analysis[:16],
    "heading_optimization": [
      {
        "current": _as_str(h.get("current")),
        "suggested": _as_str(h.get("suggested")),
        "reason": _as_str(h.get("reason")),
        "apply": _as_str(h.get("action")).lower() in {"improve", "merge", "split"},
      }
      for h in heading_analysis[:12]
    ],
    "search_intent": {
      "primary": _as_str(understanding.get("search_intent")) or "Informational",
      "article_type": article_type,
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
    "image_alt_text": image_alts,
    "schema_recommendations": schema_recs[:8],
    "metadata": metadata,
    "optimization_plan": plan,
    "conditional_rewrite": {**conditional, **rewrite_info},
    "quality_validation": validation,
    "ai_search_optimization": ai_search_lines[:12],
    "final_metrics": {
      "seo_score_before": seo_before,
      "seo_score_after": seo_after,
      "readability_score": optimized_metrics.get("readability_score"),
      "keyword_coverage_pct": None,
      "entity_coverage_pct": round(
        100 * sum(1 for e in entities[:12] if e.lower() in content.lower()) / max(1, min(len(entities), 12)),
        1,
      ),
      "content_depth_score": min(100, round(optimized_metrics.get("word_count", 0) / 8)),
      "topical_authority_score": conf_i,
      "rewrite_applied": bool(rewrite_info.get("patches_applied")),
      "rewrite_summary": rewrite_info.get("summary"),
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
      "mode": "optimizer_strategist_v2",
    },
    "use_rag": False,
    "generator_version": STRATEGIST_VERSION,
    "variation_seed": None,
    "rewrite_applied": bool(rewrite_info.get("patches_applied")),
    "article_understanding": understanding,
    "optimization_plan": plan,
    "quality_validation": validation,
    "conditional_rewrite": seo_report["conditional_rewrite"],
    "seo_report": seo_report,
    "architecture": {
      "flow": [
        "input_article",
        "article_understanding",
        "article_type_detection",
        "topic_resolution",
        "seo_audit",
        "keyword_analysis",
        "entity_analysis",
        "content_gap_analysis",
        "heading_analysis",
        "faq_generation",
        "metadata",
        "link_optimization",
        "schema_recommendations",
        "ai_search_optimization",
        "optimization_plan",
        "conditional_rewrite",
        "quality_validation",
      ],
      "stages": {
        "topic_resolution": topic_resolution_stage,
        "article_understanding": understanding,
        "article_type": article_type,
        "entity_extraction": {"from_strategist": entities, "grouped": entities_block},
        "seo_audit": seo_audit,
        "conditional_rewrite": rewrite_info,
        "quality_validation": validation,
      },
    },
    "pipeline": {
      "keyword_analysis": {"primary": primary, "secondary": secondary, "table": kw_table},
      "entity_extraction": entities,
      "coverage_map": {},
      "gap_analysis": gap_rows,
      "source_router": {"mode": "hosted_strategist_v2"},
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
      "schema_suggestions": {"recommendations": schema_recs},
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
  apply_rewrite: bool = True,
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
    apply_conditional_rewrite=apply_rewrite,
  )
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=_SYSTEM,
    use_rag=False,
    skip_intent=True,
    skip_kb_direct_match=True,
    max_tokens=6500,
    temperature=0.25,
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
    apply_rewrite=apply_rewrite,
  )
