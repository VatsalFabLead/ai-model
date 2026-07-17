"""Route chat/completions messages to any AI tool (all modules in one chat).

Slash commands (first token of the last user message):
  /seo-content <topic>                      SEO Content Generator
  /seo-optimizer <content>                  SEO Content Optimizer
  /title-meta <topic>                       SEO Title & Meta
  /keywords <seed keyword>                  SEO Keyword Generator
  /schema type: Article name: <page name>   Schema Markup Generator
  /email-new <context>                      Email Assistant — New
  /email-reply <original> | <points>        Email Assistant — Reply
  /email-cold <company> | <offer> | <value> Email Assistant — Cold
  /plagiarism <content>                     Copyright & Plagiarism Check
  /rewrite <content>                        Plagiarism Remove & Rewrite
  /cover-letter <role> | <company> | <skills>  Professional Cover Letter
  /resume name: .. job: .. email: .. phone: .. Resume Builder
  /tools                                    List all chat tools

Fields can also be given as `key: value` lines, e.g.
  /email-new subject: Demo request
  tone: friendly
  We would love a product demo next week.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.nexus_gateway import invoke_nexus_tool
from app.services.registry import ProviderRegistry

# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------

_TOOL_ALIASES: dict[str, tuple[str, ...]] = {
  "seo_content": ("seo-content", "seo_content", "seocontent", "content", "article", "blog"),
  "seo_optimizer": ("seo-optimizer", "seo_optimizer", "optimize", "optimizer"),
  "title_meta": ("title-meta", "title_meta", "titlemeta", "meta"),
  "seo_keywords": ("seo-keywords", "seo_keywords", "keywords", "keyword"),
  "schema_markup": ("schema-markup", "schema_markup", "schema", "jsonld", "json-ld"),
  "email_new": ("email-new", "email_new", "new-email", "email"),
  "email_reply": ("email-reply", "email_reply", "reply-email", "reply"),
  "email_cold": ("email-cold", "email_cold", "cold-email", "cold"),
  "plagiarism_check": ("plagiarism-check", "plagiarism_check", "plagiarism", "plag"),
  "plagiarism_remove": ("plagiarism-remove", "plagiarism_remove", "rewrite", "remove-plagiarism"),
  "cover_letter": ("cover-letter", "cover_letter", "coverletter", "cover"),
  "resume_builder": ("resume-builder", "resume_builder", "resume", "cv"),
  "help": ("tools", "help", "commands"),
}

_ALIAS_TO_TOOL: dict[str, str] = {
  alias: tool for tool, aliases in _TOOL_ALIASES.items() for alias in aliases
}

# Backend prefixes handled by backend_router — never treat as tool commands.
_BACKEND_PREFIXES = frozenset({"custom", "gemma", "ollama", "llm", "auto"})

_SLASH_RE = re.compile(r"^/([a-z0-9_-]+)\b[:\s,-]*(.*)$", re.IGNORECASE | re.DOTALL)

_FIELD_KEYS = {
  "subject": "subject",
  "tone": "tone",
  "keywords": "keywords",
  "words": "word_count",
  "word_count": "word_count",
  "audience": "audience",
  "category": "category",
  "language": "language",
  "variations": "variations",
  "type": "schema_type",
  "schema_type": "schema_type",
  "name": "name",
  "company": "company_name",
  "company_name": "company_name",
  "offer": "purpose_offer",
  "purpose": "purpose_offer",
  "purpose_offer": "purpose_offer",
  "value": "value_proposition",
  "value_proposition": "value_proposition",
  "original": "original_email",
  "original_email": "original_email",
  "points": "reply_points",
  "reply_points": "reply_points",
  "role": "job_role",
  "job_role": "job_role",
  "job": "job_title",
  "job_title": "job_title",
  "title": "job_title",
  "skills": "skills",
  "skills_experience": "skills",
  "email": "email",
  "phone": "phone",
  "linkedin": "linkedin",
  "portfolio": "portfolio",
  "education": "education",
  "experience": "experience",
  "summary": "summary",
  "projects": "projects",
  "certifications": "certifications",
  "achievements": "achievements",
  "template": "template",
  "full_name": "full_name",
  "applicant": "applicant_name",
  "applicant_name": "applicant_name",
  "content": "content",
  "topic": "topic",
  "seed": "seed_keyword",
  "seed_keyword": "seed_keyword",
  "context": "context",
  "brief": "context",
  "article_title": "title",
  "page_title": "title",
  "primary_topic": "primary_topic",
  "country": "country",
  "market": "market",
}

_KEY_LINE_RE = re.compile(
  r"^\s*(" + "|".join(sorted(_FIELD_KEYS, key=len, reverse=True)) + r")\s*[:=]\s*(.*)$",
  re.IGNORECASE,
)

# Only these fields may span multiple lines; anything after a short field
# (tone, subject, ...) is treated as free text.
_MULTILINE_FIELDS = frozenset({
  "content", "original_email", "reply_points", "purpose_offer",
  "value_proposition", "skills", "experience", "education", "summary",
  "projects", "certifications", "achievements", "topic", "seed_keyword", "context", "content",
})


_INLINE_KEY_RE = re.compile(
  r"\s+(" + "|".join(sorted(_FIELD_KEYS, key=len, reverse=True)) + r")\s*[:=]\s*",
  re.IGNORECASE,
)


def _parse_fields(text: str) -> tuple[dict[str, str], str]:
  """Split `key: value` lines from free text. Long fields may span lines."""
  fields: dict[str, str] = {}
  free: list[str] = []
  current_key: str | None = None
  for line in (text or "").splitlines():
    m = _KEY_LINE_RE.match(line)
    if m:
      current_key = _FIELD_KEYS[m.group(1).lower()]
      value = m.group(2).strip()
      if current_key in _MULTILINE_FIELDS:
        fields[current_key] = value
      else:
        # Short fields may share a line: "type: Article name: My Page"
        parts = _INLINE_KEY_RE.split(value)
        fields[current_key] = parts[0].strip()
        current_key = None
        for i in range(1, len(parts), 2):
          key = _FIELD_KEYS[parts[i].lower()]
          fields[key] = parts[i + 1].strip()
          current_key = key if key in _MULTILINE_FIELDS else None
    elif current_key is not None and line.strip():
      fields[current_key] = (fields[current_key] + "\n" + line.rstrip()).strip()
    else:
      current_key = None
      free.append(line)
  return fields, "\n".join(free).strip()


def _split_pipes(text: str, n: int) -> list[str]:
  parts = [p.strip() for p in text.split("|")]
  parts = [p for p in parts if p]
  while len(parts) < n:
    parts.append("")
  return parts[:n]


def parse_tool_command(text: str) -> tuple[str, str] | None:
  """Return (tool_id, args_text) when the message starts with a tool command."""
  m = _SLASH_RE.match((text or "").strip())
  if not m:
    return None
  cmd = m.group(1).lower()
  if cmd in _BACKEND_PREFIXES:
    return None
  tool = _ALIAS_TO_TOOL.get(cmd)
  if not tool:
    return None
  return tool, m.group(2).strip()


# ---------------------------------------------------------------------------
# Natural-language intent detection (conservative)
# ---------------------------------------------------------------------------

_NL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
  ("cover_letter", re.compile(r"\bcover\s+letter\b", re.IGNORECASE)),
  ("resume_builder", re.compile(r"\b(build|create|make|generate|write)\b[^.\n]*\bresume\b|\bresume\s+builder\b", re.IGNORECASE)),
  ("plagiarism_remove", re.compile(r"\b(remove|fix|reduce)\b[^.\n]*\bplagiarism\b", re.IGNORECASE)),
  ("plagiarism_check", re.compile(r"\bplagiarism\b|\boriginality\s+check\b", re.IGNORECASE)),
  ("email_cold", re.compile(r"\bcold\s+email\b", re.IGNORECASE)),
  ("email_reply", re.compile(r"\breply\b[^.\n]*\bemail\b|\bemail\s+reply\b", re.IGNORECASE)),
  ("email_new", re.compile(r"\b(write|draft|compose)\b[^.\n]*\be?-?mail\b", re.IGNORECASE)),
  ("seo_keywords", re.compile(r"\b(seo\s+)?keywords?\b[^.\n]*\b(for|about|ideas?)\b|\bkeyword\s+(generator|research)\b", re.IGNORECASE)),
  ("schema_markup", re.compile(r"\bschema\s+markup\b|\bjson-?ld\b", re.IGNORECASE)),
  ("seo_optimizer", re.compile(r"\boptimi[sz]e\b[^.\n]*\b(content|article|text|blog|page|seo)\b", re.IGNORECASE)),
  ("seo_content", re.compile(
    r"\b(write|create|generate)\b[^.\n]*\b(blog|article|seo\s+content)\b"
    r"|\bseo\s+content\b"
    r"|\b(blog\s+post|article)\s+(about|on|for)\b",
    re.IGNORECASE,
  )),
)


def detect_tool_intent(text: str) -> str | None:
  t = (text or "").strip()
  if not t or t.startswith("/"):
    return None
  for tool, pattern in _NL_PATTERNS:
    if pattern.search(t):
      return tool
  return None


# ---------------------------------------------------------------------------
# Usage / help
# ---------------------------------------------------------------------------

_USAGE: dict[str, str] = {
  "seo_content": "Usage: `/seo-content <topic>` — optional `tone:`, `keywords:`, `words:`, `audience:`, `language:`. Returns article + slug + suggested tags.",
  "seo_optimizer": "Usage: `/seo-optimizer <paste your content>` — optional `keywords:`, `tone:`.",
  "title_meta": "Usage: `/title-meta <topic>` — optional `variations:` (10–50).",
  "seo_keywords": "Usage: `/keywords title: <title> content: <article>` — or `context: <brief>`. Optional `primary_topic:`, `country:`, `language:`, `variations:`. With Use AI / hosted LLM runs the SEO strategist (JSON groups).",
  "schema_markup": "Usage: `/schema type: Article name: <page or business name>` — optional `language:`. Types: Article, Product, FAQPage, LocalBusiness, Recipe, JobPosting, …",
  "email_new": "Usage: `/email-new <context / key points>` — optional `subject:`, `tone:` (professional | casual | friendly | formal).",
  "email_reply": "Usage: `/email-reply <original email> | <your reply points>` — optional `tone:`. You can also use `original:` and `points:` fields.",
  "email_cold": "Usage: `/email-cold <company> | <purpose / offer> | <value proposition>` — optional `tone:`. Or use `company:`, `offer:`, `value:` fields.",
  "plagiarism_check": "Usage: `/plagiarism <paste at least 40 characters of content>`.",
  "plagiarism_remove": "Usage: `/rewrite <paste at least 40 characters of content>` — rewrites to reduce similarity.",
  "cover_letter": "Usage: `/cover-letter <job role> | <company> | <skills & experience>` — optional `tone:`, `applicant:`. Or use `role:`, `company:`, `skills:` fields.",
  "resume_builder": (
    "Usage: `/resume` with fields:\n"
    "```\n/resume name: Vatsal Patel\njob: Flutter Developer\nemail: you@example.com\nphone: +91-9876543210\n"
    "skills: Flutter, Dart, Firebase\nexperience: Built 3 production apps\neducation: B.Tech CS, 2022\n```\n"
    "Required: `name`, `job`, `email`, `phone`. Optional: `skills`, `experience`, `education`, `summary`, `projects`, `template`, …"
  ),
}

_HELP_TEXT = (
  "## AI Tools available in this chat\n\n"
  "| Command | Tool |\n|---|---|\n"
  "| `/seo-content <topic>` | SEO Content Generator |\n"
  "| `/seo-optimizer <content>` | SEO Content Optimizer |\n"
  "| `/title-meta <topic>` | SEO Title & Meta |\n"
  "| `/keywords context: <brief>` | SEO Keyword Generator |\n"
  "| `/schema type: Article name: …` | Schema Markup Generator |\n"
  "| `/email-new <context>` | Email Assistant — New |\n"
  "| `/email-reply <original> \\| <points>` | Email Assistant — Reply |\n"
  "| `/email-cold <company> \\| <offer> \\| <value>` | Email Assistant — Cold |\n"
  "| `/plagiarism <content>` | Copyright & Plagiarism Check |\n"
  "| `/rewrite <content>` | Plagiarism Remove & Rewrite |\n"
  "| `/cover-letter <role> \\| <company> \\| <skills>` | Professional Cover Letter |\n"
  "| `/resume name: … job: … email: … phone: …` | Resume Builder |\n\n"
  "You can also just ask naturally, e.g. *\"write an email about the project delay\"* "
  "or *\"generate keywords for food delivery apps\"*."
)


# ---------------------------------------------------------------------------
# Input builders — turn chat text into nexus tool input, or usage reply
# ---------------------------------------------------------------------------


def _build_input(tool: str, args_text: str) -> dict[str, Any] | str:
  """Return input_data for invoke_nexus_tool, or a usage string when incomplete."""
  fields, free = _parse_fields(args_text)

  if tool == "seo_content":
    topic = fields.get("topic") or free
    if not topic:
      return _USAGE[tool]
    data: dict[str, Any] = {"topic": topic}
    if fields.get("keywords"):
      data["keywords"] = fields["keywords"]
    if fields.get("tone"):
      data["tone"] = fields["tone"].lower()
    return data

  if tool == "seo_optimizer":
    content = fields.get("content") or free
    if not content:
      return _USAGE[tool]
    data = {"content": content, "use_ai": True, "rewrite": False, "mode": "strategist"}
    if fields.get("keywords"):
      data["keywords"] = fields["keywords"]
    if fields.get("tone"):
      data["tone"] = fields["tone"].lower()
    if fields.get("rewrite") in ("1", "true", "yes", "rewrite"):
      data["rewrite"] = True
      data["mode"] = "rewrite"
    return data

  if tool == "title_meta":
    topic = fields.get("topic") or free
    if not topic:
      return _USAGE[tool]
    data = {"topic": topic}
    if fields.get("variations"):
      data["variations"] = _to_int(fields["variations"], 10, 10, 50)
    return data

  if tool == "seo_keywords":
    context = fields.get("context") or ""
    content = fields.get("content") or ""
    title = fields.get("title") or ""
    seed = fields.get("seed_keyword") or fields.get("topic") or fields.get("primary_topic") or ""
    if not context and not content and not seed and not title and free:
      if len(free) > 120 or free.count(" ") >= 18:
        content = free
      else:
        seed = free
    if not seed and not context and not content and not title:
      return _USAGE[tool]
    data: dict[str, Any] = {}
    if seed:
      data["seed_keyword"] = seed
      data["primary_topic"] = fields.get("primary_topic") or seed
    if title:
      data["title"] = title
    if content:
      data["content"] = content
    if context:
      data["context"] = context
    if fields.get("country") or fields.get("market"):
      data["country"] = fields.get("country") or fields.get("market")
    if fields.get("language"):
      data["language"] = fields["language"]
    if fields.get("variations"):
      data["variations"] = _to_int(fields["variations"], 10, 10, 50)
    data["use_ai"] = True
    data["mode"] = "strategist" if (content or context or title) else "pipeline"
    return data

  if tool == "schema_markup":
    schema_type = fields.get("schema_type", "")
    name = fields.get("name") or free
    if not schema_type and free:
      # Allow "/schema Article My Page Name"
      first, _, rest = free.partition(" ")
      if first and rest:
        schema_type, name = first, rest.strip()
    if not schema_type or not name:
      return _USAGE[tool]
    data = {"schema_type": schema_type, "name": name}
    if fields.get("language"):
      data["language"] = fields["language"]
    return data

  if tool == "email_new":
    context = fields.get("content") or free
    if not context:
      return _USAGE[tool]
    return {
      "subject": fields.get("subject", ""),
      "context": context,
      "tone": (fields.get("tone") or "professional").lower(),
    }

  if tool == "email_reply":
    original = fields.get("original_email", "")
    points = fields.get("reply_points", "")
    if not original and free:
      original, points_from_pipe = (_split_pipes(free, 2) if "|" in free else (free, ""))
      points = points or points_from_pipe
    if not original:
      return _USAGE[tool]
    return {
      "original_email": original,
      "reply_points": points,
      "tone": (fields.get("tone") or "professional").lower(),
    }

  if tool == "email_cold":
    company = fields.get("company_name", "")
    offer = fields.get("purpose_offer", "")
    value = fields.get("value_proposition", "")
    if not (company and offer and value) and free:
      p = _split_pipes(free, 3)
      company, offer, value = company or p[0], offer or p[1], value or p[2]
    if not (company and offer and value):
      return _USAGE[tool]
    return {
      "company_name": company,
      "purpose_offer": offer,
      "value_proposition": value,
      "tone": (fields.get("tone") or "professional").lower(),
    }

  if tool in ("plagiarism_check", "plagiarism_remove"):
    content = fields.get("content") or free
    if len(content) < 40:
      return _USAGE[tool]
    return {"content": content}

  if tool == "cover_letter":
    role = fields.get("job_role") or fields.get("job_title", "")
    company = fields.get("company_name", "")
    skills = fields.get("skills", "")
    if not (role and company and skills) and free:
      p = _split_pipes(free, 3)
      role, company, skills = role or p[0], company or p[1], skills or p[2]
    if not (role and company and skills):
      return _USAGE[tool]
    data = {"job_role": role, "company_name": company, "skills_experience": skills}
    if fields.get("tone"):
      data["tone"] = fields["tone"].lower()
    if fields.get("applicant_name"):
      data["applicant_name"] = fields["applicant_name"]
    if fields.get("language"):
      data["language"] = fields["language"]
    return data

  if tool == "resume_builder":
    required = ("full_name", "job_title", "email", "phone")
    aliases = {"full_name": fields.get("full_name") or fields.get("name", "")}
    data = {
      "full_name": aliases["full_name"],
      "job_title": fields.get("job_title", ""),
      "email": fields.get("email", ""),
      "phone": fields.get("phone", ""),
    }
    if not all(data[k] for k in required):
      return _USAGE[tool]
    for key in (
      "linkedin", "portfolio", "education", "experience", "skills", "summary",
      "projects", "certifications", "achievements", "template", "language",
    ):
      if fields.get(key):
        data[key] = fields[key]
    return data

  return _USAGE.get(tool, _HELP_TEXT)


def _to_int(raw: str, default: int, lo: int, hi: int) -> int:
  try:
    return max(lo, min(hi, int(raw)))
  except (TypeError, ValueError):
    return default


# ---------------------------------------------------------------------------
# Reply formatting — structured tool result -> markdown chat message
# ---------------------------------------------------------------------------


def _fmt_seo_content(r: dict[str, Any]) -> str:
  meta = r.get("metadata") or {}
  quality = r.get("quality") or {}
  kw = r.get("keywords") or {}
  primary = kw.get("primary", "") if isinstance(kw, dict) else ""
  secondary = (kw.get("secondary") or []) if isinstance(kw, dict) else []
  tags = r.get("suggested_tags") or []
  slug = r.get("slug") or ""
  lines = [
    f"## {meta.get('title') or r.get('title') or r.get('topic', 'SEO Content')}",
    f"**Meta description:** {meta.get('meta_description') or r.get('meta_description', '')}",
    f"**Slug:** /{slug}" if slug else "**Slug:** —",
    f"**Suggested tags:** {', '.join(tags[:12])}" if tags else "**Suggested tags:** —",
    f"**Keywords:** {', '.join([primary] + list(secondary)[:5]).strip(', ')}",
    "",
    r.get("article") or (r.get("content") or {}).get("article", ""),
    "",
    f"_SEO score: {quality.get('seo_score', 0)}/100 · Words: {r.get('word_count', 0)}_",
  ]
  return "\n".join(lines).strip()


def _fmt_seo_optimizer(r: dict[str, Any]) -> str:
  report = r.get("seo_report") or (r.get("optimization") or {}).get("seo_report") or {}
  understanding = r.get("article_understanding") or report.get("article_understanding") or {}
  plan = r.get("optimization_plan") or report.get("optimization_plan") or {}
  lines = [
    f"## SEO Content Optimizer · v{r.get('generator_version', '?')}",
    f"**Mode:** {(r.get('ai') or {}).get('mode') or ('rewrite' if r.get('rewrite_applied') else 'strategist')}",
    f"**SEO score:** {r.get('seo_score_before', 0)} → {r.get('seo_score_after', 0)} "
    f"(+{r.get('improvement', 0)})",
    "",
  ]
  if understanding:
    lines += [
      "### Article Understanding",
      f"- Primary: {understanding.get('primary_topic') or (report.get('keyword_analysis') or {}).get('primary') or '—'}",
      f"- Type: {understanding.get('article_type', '—')} · Intent: {understanding.get('search_intent', '—')}",
      f"- Industry: {understanding.get('industry', '—')} · Audience: {understanding.get('target_audience', '—')}",
      "",
    ]
  kw = report.get("keyword_analysis") or {}
  if kw.get("primary"):
    lines += [
      "### Keywords",
      f"- Primary: {kw.get('primary')}",
      f"- Secondary: {', '.join(kw.get('secondary') or []) or '—'}",
      "",
    ]
  meta = (report.get("metadata") or (r.get("optimization") or {}).get("metadata") or {})
  if meta.get("title") or meta.get("meta_description"):
    lines += [
      "### Metadata",
      f"- Title: {meta.get('title', '—')}",
      f"- Meta: {meta.get('meta_description', '—')}",
      "",
    ]
  for bucket, label in (
    ("high_priority", "High"),
    ("medium_priority", "Medium"),
    ("low_priority", "Low"),
  ):
    items = plan.get(bucket) or []
    if items:
      if "### Optimization Plan" not in lines:
        lines.append("### Optimization Plan")
      for p in items[:5]:
        if isinstance(p, dict):
          lines.append(f"- [{label}] {p.get('action', '')}" + (f" — {p['why']}" if p.get("why") else ""))
        else:
          lines.append(f"- [{label}] {p}")
  if "### Optimization Plan" in lines:
    lines.append("")
  faqs = report.get("faqs") or (r.get("optimization") or {}).get("faqs") or []
  if faqs:
    lines.append("### FAQs")
    for i, f in enumerate(faqs[:6], 1):
      if isinstance(f, dict):
        lines.append(f"{i}. {f.get('question', '')}")
        lines.append(f.get("answer", ""))
      else:
        lines.append(f"{i}. {f}")
    lines.append("")
  suggestions = r.get("suggestions") or []
  if suggestions:
    lines += ["### Suggestions"] + [f"- {s}" for s in suggestions[:8]] + [""]
  body_label = "Optimized Article" if r.get("rewrite_applied") else "Original Article (preserved)"
  lines += [f"### {body_label}", r.get("optimized_content", "")]
  return "\n".join(lines).strip()


def _fmt_title_meta(r: dict[str, Any]) -> str:
  from app.engine.chat_intents import format_title_meta_reply

  return format_title_meta_reply(r)


def _fmt_seo_keywords(r: dict[str, Any]) -> str:
  lines = [
    f"## SEO Keywords — {r.get('seed_keyword', '')}",
    f"_v{r.get('generator_version', '?')} · {r.get('count', 0)} keywords_",
    "",
  ]
  if r.get("title"):
    lines.append(f"**Title:** {r['title']}")
  summary = r.get("summary") or {}
  if summary.get("content_type") or summary.get("industry"):
    lines.append(
      f"**Type:** {summary.get('content_type', '—')} · "
      f"**Intent:** {summary.get('search_intent', '—')} · "
      f"**Industry:** {summary.get('industry', '—')}"
    )
  ctx = (r.get("context") or "").strip()
  if ctx and ctx != (r.get("seed_keyword") or ""):
    preview = ctx if len(ctx) <= 280 else ctx[:277] + "…"
    lines += [f"**Content:** {preview}", ""]
  elif lines[-1] != "":
    lines.append("")
  arch = r.get("architecture") or {}
  flow = arch.get("flow") or []
  if flow:
    lines += ["### Pipeline stages", " → ".join(flow), ""]
  stages = arch.get("stages") or {}
  topic = (stages.get("topic_extraction") or {}).get("primary_topic")
  if not topic:
    topics = (stages.get("topic_extraction") or {}).get("topics") or (r.get("pipeline") or {}).get("topics") or []
    topic = topics[0] if topics else None
  industry = stages.get("industry_classification") or summary.get("industry")
  if topic or industry:
    ind = industry.get("industry") if isinstance(industry, dict) else industry
    lines.append(f"**Topic:** {topic or '—'} · **Industry:** {ind or '—'}")
    lines.append("")
  out = r.get("output") or {}
  if out.get("topics"):
    lines += ["### Topics", ", ".join(str(t) for t in out["topics"][:12]), ""]
  gaps = r.get("content_gap_analysis") or out.get("content_gap_analysis") or {}
  if gaps.get("missing_topics") or gaps.get("recommended_sections"):
    lines += ["### Content gaps"]
    if gaps.get("missing_topics"):
      lines.append("- Missing: " + "; ".join(str(x) for x in gaps["missing_topics"][:6]))
    if gaps.get("recommended_sections"):
      lines.append("- Add sections: " + "; ".join(str(x) for x in gaps["recommended_sections"][:5]))
    lines.append("")
  serp = r.get("serp_predictions") or out.get("serp_predictions") or []
  if serp:
    lines.append("### SERP predictions")
    for s in serp[:5]:
      feats = ", ".join(s.get("likely_serp_features") or []) or "—"
      lines.append(f"- **{s.get('keyword', '')}** → {feats}")
    lines.append("")
  clusters = r.get("topic_clusters") or {}
  if clusters:
    lines.append("### Clusters")
    for name, kws in list(clusters.items())[:8]:
      sample = ", ".join(kws[:5]) if isinstance(kws, list) else ""
      lines.append(f"- **{name}:** {sample}")
    lines.append("")
  lines += [
    "| # | Keyword | Intent | Volume | Difficulty | Opp |",
    "|---|---------|--------|--------|------------|-----|",
  ]
  for i, kw in enumerate((r.get("keywords") or [])[:50], 1):
    lines.append(
      f"| {i} | {kw.get('keyword', '')} | {kw.get('intent', '')} | {kw.get('volume_estimate', '')} "
      f"| {kw.get('difficulty_estimate', '')} | {kw.get('opportunity_score', '')} |"
    )
  recs = r.get("recommendations") or []
  if recs:
    lines += ["", "### Recommendations"] + [f"- {s}" for s in recs[:5]]
  return "\n".join(lines).strip()


def _fmt_schema_markup(r: dict[str, Any]) -> str:
  jsonld = r.get("jsonld_string") or json.dumps(r.get("jsonld") or {}, indent=2, ensure_ascii=False)
  quality = r.get("quality") or {}
  return (
    f"## Schema Markup — {r.get('schema_type', '')}\n\n"
    f"```json\n{jsonld}\n```\n\n"
    f"_Completeness: {quality.get('completeness_score', 0)}/100 · "
    f"SEO-ready: {'yes' if quality.get('seo_ready') else 'no'}_"
  )


def _fmt_email(r: dict[str, Any]) -> str:
  quality = r.get("quality") or {}
  overall = quality.get("overall") if isinstance(quality, dict) else 0
  lines = [
    f"**Subject:** {r.get('subject', '')}",
    "",
    r.get("email", ""),
  ]
  if overall:
    lines += ["", f"_Quality: {overall}/100 · Tone: {r.get('tone', '')} · Words: {r.get('word_count', 0)}_"]
  suggestions = r.get("suggestions") or []
  if suggestions:
    lines += ["", "### Suggestions"] + [f"- {s}" for s in suggestions[:5]]
  return "\n".join(lines).strip()


def _fmt_plagiarism(r: dict[str, Any]) -> str:
  lines = [
    "## Plagiarism Report",
    f"**Similarity:** {r.get('similarity_percent', 0)}% · "
    f"**Original:** {r.get('original_percent', 100)}% · "
    f"**Risk:** {r.get('risk_level', 'low')}",
    "",
    r.get("summary", ""),
  ]
  flags = r.get("flags") or []
  if flags:
    lines += ["", "### Flags"] + [f"- {f}" for f in flags[:6]]
  suggestions = r.get("suggestions") or []
  if suggestions:
    lines += ["", "### Suggestions"] + [f"- {s}" for s in suggestions[:6]]
  return "\n".join(lines).strip()


def _fmt_plagiarism_remove(r: dict[str, Any]) -> str:
  return (
    "## Rewritten Content\n\n"
    f"**Similarity:** {r.get('similarity_percent_before', 0)}% → "
    f"{r.get('similarity_percent_after', 0)}% "
    f"(improvement {r.get('improvement', 0)})\n\n"
    f"{r.get('rewritten_content', '')}"
  )


def _fmt_cover_letter(r: dict[str, Any]) -> str:
  quality = r.get("quality") or {}
  return (
    f"## Cover Letter — {r.get('job_role', '')} at {r.get('company_name', '')}\n\n"
    f"{r.get('cover_letter_markdown') or r.get('cover_letter', '')}\n\n"
    f"_Quality: {quality.get('quality_score', 0)}/100 · Words: {quality.get('word_count', 0)} · "
    f"ATS coverage: {quality.get('ats_coverage_pct', 0)}%_"
  )


def _fmt_resume(r: dict[str, Any]) -> str:
  quality = r.get("quality") or {}
  return (
    f"{r.get('resume_markdown') or r.get('resume_ai_text', '')}\n\n"
    f"_Completeness: {quality.get('completeness_score', 0)}/100 · "
    f"Template: {r.get('template', 'modern')}_"
  )


_FORMATTERS = {
  "seo_content": _fmt_seo_content,
  "seo_optimizer": _fmt_seo_optimizer,
  "title_meta": _fmt_title_meta,
  "seo_keywords": _fmt_seo_keywords,
  "schema_markup": _fmt_schema_markup,
  "email_new": _fmt_email,
  "email_reply": _fmt_email,
  "email_cold": _fmt_email,
  "plagiarism_check": _fmt_plagiarism,
  "plagiarism_remove": _fmt_plagiarism_remove,
  "cover_letter": _fmt_cover_letter,
  "resume_builder": _fmt_resume,
}


# ---------------------------------------------------------------------------
# Entry point used by /chat/completions
# ---------------------------------------------------------------------------


async def maybe_handle_tool(
  registry: ProviderRegistry,
  messages: list[dict[str, str]],
) -> tuple[str, str] | None:
  """If the last user message targets a tool, run it.

  Returns (reply_markdown, backend_label) or None to continue with normal chat.
  """
  last_user = next(
    (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
    "",
  )
  parsed = parse_tool_command(last_user)
  if parsed:
    tool, args_text = parsed
  else:
    tool = detect_tool_intent(last_user)
    if not tool:
      return None
    args_text = last_user

  if tool == "help":
    return _HELP_TEXT, "tool:help"

  built = _build_input(tool, args_text)
  if isinstance(built, str):
    # Missing required fields — reply with usage instructions.
    return built, f"tool:{tool}"

  try:
    envelope = await invoke_nexus_tool(registry, tool=tool, input_data=built)
  except ValueError as exc:
    return f"{exc}\n\n{_USAGE.get(tool, '')}".strip(), f"tool:{tool}"

  result = envelope.get("result") or {}
  formatter = _FORMATTERS.get(tool)
  reply = formatter(result) if formatter else json.dumps(result, indent=2, ensure_ascii=False)
  return reply, f"tool:{tool}"
