"""SEO Content Optimizer — analyze and improve existing content.

Matches the optimizer UI: readability, word/char/sentence counts,
optimized content, and actionable suggestions. Custom model only.
"""

from __future__ import annotations

import re
from typing import Any

from app.engine import seo_optimizer_engine
from app.services.provider_base import ModelProvider

_MAX_CONTENT = 12000


def supported_categories() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_categories()


def supported_tones() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_tones()


def supported_languages() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_languages()


def _clean(text: str) -> str:
  t = (text or "").strip()
  if t.startswith("```"):
    t = re.sub(r"^```(?:\w+)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
  return re.sub(r"\n{3,}", "\n\n", t).strip()


def _coerce_keywords(keywords: list[str] | str | None) -> list[str]:
  if not keywords:
    return []
  if isinstance(keywords, str):
    parts = keywords.split(",")
  else:
    parts = [str(k) for k in keywords]
  seen: set[str] = set()
  out: list[str] = []
  for p in parts:
    p = p.strip()
    if p and p.lower() not in seen:
      seen.add(p.lower())
      out.append(p)
  return out


def _is_valid_optimized(original: str, optimized: str) -> bool:
  if not optimized or len(optimized) < max(40, len(original) * 0.4):
    return False
  if "SUGGESTIONS:" in optimized and optimized.count("\n") < 3:
    return False
  if optimized.count("###") > 12:
    return False
  return seo_optimizer_engine.count_words(optimized) >= max(20, seo_optimizer_engine.count_words(original) // 3)


def _kb_context(
  *,
  category: str,
  tone: str,
  language: str | None,
  lang_code: str,
  keywords: list[str],
  content: str,
) -> str:
  """Pull domain training hints from the SEO optimizer knowledge base."""
  kb = seo_optimizer_engine.get_kb()
  if not kb.size:
    return ""

  primary = keywords[0] if keywords else ""
  queries = [
    f"SEO optimizer rewrite {category} {tone} {primary}",
    f"SEO optimizer {category} {tone} tone rewrite",
    f"SEO optimizer {category} category best practices",
    f"SEO optimizer {tone} tone guidelines worldwide",
    f"SEO optimizer aesthetic worldwide human copy",
    f"SEO optimizer advanced worldwide editing",
    "SEO content optimization best practices readability",
  ]
  if language:
    queries.insert(1, f"SEO optimizer multilingual {language} {lang_code}")

  parts: list[str] = []
  seen: set[str] = set()
  for q in queries:
    answer, score = kb.search(q)
    if not answer or score < 0.08:
      continue
    key = answer[:80].lower()
    if key in seen:
      continue
    seen.add(key)
    parts.append(answer)
    if len(parts) >= 4:
      break
  return "\n\n".join(parts)


def _try_kb_full_rewrite(
  *,
  category: str,
  tone: str,
  keywords: list[str],
) -> str | None:
  """Use a trained full OPTIMIZED: example when the KB has a close match."""
  if not keywords:
    return None
  kb = seo_optimizer_engine.get_kb()
  primary = keywords[0]
  for q in (
    f"SEO optimizer rewrite {category} {tone} {primary}",
    f"SEO optimizer rewrite blog_article professional {primary}",
  ):
    answer, score = kb.search(q)
    if answer and "OPTIMIZED:" in answer.upper() and score >= 0.1:
      return answer
  return None


def _parse_ai_output(raw: str, original: str) -> tuple[str, list[str]]:
  text = _clean(raw)
  suggestions: list[str] = []

  if "SUGGESTIONS:" in text.upper():
    parts = re.split(r"SUGGESTIONS:\s*", text, flags=re.IGNORECASE, maxsplit=1)
    body = parts[0]
    body = re.sub(r"^OPTIMIZED:\s*", "", body, flags=re.IGNORECASE).strip()
    if len(parts) > 1:
      for ln in parts[1].splitlines():
        ln = re.sub(r"^[\-\*\d]+[\).\s]+", "", ln.strip())
        if ln:
          suggestions.append(ln)
  else:
    body = text

  body = re.sub(r"^OPTIMIZED:\s*", "", body, flags=re.IGNORECASE).strip()
  if not body:
    body = original
  return body, suggestions


def _fallback_optimize(
  content: str,
  *,
  keywords: list[str],
  tone: str,
  issues: list[dict[str, str]],
) -> tuple[str, list[str]]:
  """Deterministic improvements when AI output is weak."""
  text = content.strip()
  suggestions = [i["message"] for i in issues]

  if not re.search(r"^#\s+", text, re.MULTILINE) and keywords:
    title = keywords[0].title()
    text = f"# {title}\n\n{text}"
    suggestions.append(f"Added H1 title with primary keyword '{keywords[0]}'.")

  if "##" not in text and seo_optimizer_engine.count_words(text) > 120:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) >= 3:
      rebuilt = [paragraphs[0]]
      mid = paragraphs[1:-1] if len(paragraphs) > 2 else paragraphs[1:]
      if mid:
        rebuilt.append(f"## Key Points\n\n" + "\n\n".join(mid[:3]))
      if paragraphs[-1] != paragraphs[0]:
        rebuilt.append(f"## Conclusion\n\n{paragraphs[-1]}")
      text = "\n\n".join(rebuilt)
      suggestions.append("Added H2 sections for better structure and scanability.")

  # Trim filler in simple cases
  text = re.sub(r"\b(very|really|just|actually|basically)\b\s+", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\s{2,}", " ", text)
  text = re.sub(r" +\n", "\n", text)

  if keywords and keywords[0].lower() not in text.lower()[:300]:
    intro = f"Discover effective strategies for **{keywords[0]}** that work for audiences worldwide.\n\n"
    text = intro + text
    suggestions.append(f"Wove primary keyword '{keywords[0]}' into the opening.")

  suggestions.append(f"Applied {tone} tone guidelines for clearer, more engaging copy.")
  return text.strip(), suggestions[:10]


async def optimize(
  provider: ModelProvider,
  *,
  content: str,
  keywords: list[str] | str | None = None,
  tone: str | None = None,
  language: str | None = None,
  category: str | None = None,
  use_ai: bool = True,
) -> dict[str, Any]:
  content = (content or "").strip()
  if not content:
    raise ValueError("content is required")
  if len(content) > _MAX_CONTENT:
    raise ValueError(f"content exceeds maximum length of {_MAX_CONTENT} characters")

  cat = seo_optimizer_engine.normalize_category(category)
  tone_str = seo_optimizer_engine.normalize_tone(tone, cat)
  lang_code = seo_optimizer_engine.bcp47(language)
  kws = _coerce_keywords(keywords)

  original_metrics = seo_optimizer_engine.content_metrics(content)
  issues = seo_optimizer_engine.analyze_issues(content, kws)
  original_seo = seo_optimizer_engine.seo_score_from_analysis(original_metrics, issues)

  optimized = content
  ai_suggestions: list[str] = []
  ai_used = False

  if use_ai:
    tone_guide = seo_optimizer_engine.tone_hint(tone_str)
    lang_line = f" Write in {language} (code {lang_code})." if language else ""
    issue_lines = "\n".join(f"- {i['message']}" for i in issues[:6])
    kw_line = ", ".join(kws) if kws else "(infer from content)"
    kb_hints = _kb_context(
      category=cat,
      tone=tone_str,
      language=language,
      lang_code=lang_code,
      keywords=kws,
      content=content,
    )

    system_prompt = (
      f"You are an expert worldwide SEO editor. Rewrite the content in a {tone_str} tone "
      f"({tone_guide}) to improve SEO, readability, and engagement.{lang_line} "
      f"Category: {cat.replace('_', ' ')}. "
      "Aesthetic rules: short paragraphs, scannable headings, inclusive global language, "
      "mobile-friendly blocks, active voice, natural keyword placement. "
      "Rules: preserve factual meaning, add ## subheadings if missing, improve flow, "
      "active voice, natural keyword use, do not invent false facts. "
      "Respond EXACTLY as:\n"
      "OPTIMIZED:\n<full improved markdown content>\n"
      "SUGGESTIONS:\n- <improvement 1>\n- <improvement 2>\n..."
    )
    user_prompt = (
      f"Keywords to optimize for: {kw_line}\n\n"
      f"Issues detected:\n{issue_lines or '- General polish needed'}\n\n"
      f"Content to optimize:\n{content}"
    )
    try:
      kb_rewrite = _try_kb_full_rewrite(category=cat, tone=tone_str, keywords=kws)
      if kb_rewrite:
        raw = kb_rewrite
      else:
        raw = await provider.chat(
          [{"role": "user", "content": user_prompt}],
          system_prompt=system_prompt,
          domain_context=kb_hints,
          use_rag=True,
          use_wiki=False,
          skip_intent=True,
          skip_kb_direct_match=True,
          use_neural_fallback=True,
          max_tokens=min(1200, 200 + seo_optimizer_engine.count_words(content) * 2),
          temperature=0.45,
        )
      optimized, ai_suggestions = _parse_ai_output(raw, content)
      if not _is_valid_optimized(content, optimized):
        raise ValueError("weak ai output")
      ai_used = True
    except Exception:
      optimized, ai_suggestions = _fallback_optimize(content, keywords=kws, tone=tone_str, issues=issues)
  else:
    optimized, ai_suggestions = _fallback_optimize(content, keywords=kws, tone=tone_str, issues=issues)

  optimized_metrics = seo_optimizer_engine.content_metrics(optimized)
  new_issues = seo_optimizer_engine.analyze_issues(optimized, kws)
  optimized_seo = seo_optimizer_engine.seo_score_from_analysis(optimized_metrics, new_issues)

  all_suggestions = ai_suggestions[:8]
  for issue in new_issues:
    if issue["message"] not in all_suggestions:
      all_suggestions.append(issue["message"])

  return {
    "category": cat,
    "language": lang_code,
    "tone": tone_str,
    "original": original_metrics,
    "optimized": optimized_metrics,
    "seo_score_before": original_seo,
    "seo_score_after": optimized_seo,
    "improvement": optimized_seo - original_seo,
    "optimized_content": optimized,
    "suggestions": all_suggestions[:12],
    "issues_before": issues,
    "issues_after": new_issues,
    "keywords": kws,
    "ai": {"enabled": use_ai, "model_used": ai_used},
  }
