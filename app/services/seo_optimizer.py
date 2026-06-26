"""SEO Content Optimizer — RAG pipeline + optional custom model polish.

Production flow: extract → analyze → competitor retrieval → gaps → rewrite.
Open datasets only — no GPT/Claude/Gemini.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.engine import seo_optimizer_engine
from app.engine.seo_optimizer_enrichment import parse_keywords_flexible, rewrite_content_for_keywords
from app.engine.seo_optimizer_rag_pipeline import (
  is_optimizer_instruction_content,
  normalize_pasted_optimizer_content,
  run_optimizer_rag_pipeline,
)
from app.services.provider_base import ModelProvider

logger = logging.getLogger(__name__)

_MAX_CONTENT = 12000


def supported_categories() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_categories()


def supported_tones() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_tones()


def supported_languages() -> list[dict[str, str]]:
  return seo_optimizer_engine.supported_languages()


def _coerce_keywords(keywords: list[str] | str | None) -> list[str]:
  return parse_keywords_flexible(keywords)


def _clean(text: str) -> str:
  t = (text or "").strip()
  if t.startswith("```"):
    t = re.sub(r"^```(?:\w+)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
  return re.sub(r"\n{3,}", "\n\n", t).strip()


def _parse_ai_output(raw: str, original: str) -> tuple[str, list[str]]:
  text = _clean(raw)
  suggestions: list[str] = []
  if "SUGGESTIONS:" in text.upper():
    parts = re.split(r"SUGGESTIONS:\s*", text, flags=re.I, maxsplit=1)
    body = re.sub(r"^OPTIMIZED:\s*", "", parts[0], flags=re.I).strip()
    if len(parts) > 1:
      for ln in parts[1].splitlines():
        ln = re.sub(r"^[\-\*\d]+[\).\s]+", "", ln.strip())
        if ln:
          suggestions.append(ln)
  else:
    body = text
  body = re.sub(r"^OPTIMIZED:\s*", "", body, flags=re.I).strip()
  return (body or original), suggestions


async def _enhance_with_ai(
  provider: ModelProvider,
  draft: str,
  *,
  content: str,
  keywords: list[str],
  tone: str,
  category: str,
  lang_line: str,
  evidence: str,
  suggestions: list[str],
) -> tuple[str, list[str], bool]:
  tone_guide = seo_optimizer_engine.tone_hint(tone)
  kw_line = ", ".join(keywords) if keywords else "infer from content"
  system_prompt = (
    f"You are an expert SEO editor ({tone} — {tone_guide}). "
    f"Rewrite and optimize the article for these target keywords: {kw_line}. "
    f"Category: {category}.{lang_line} "
    "Naturally weave every target keyword into headings, intro, body, and conclusion. "
    "Preserve facts from the original; improve structure, depth, and SEO. Do not invent false claims. "
    "Respond EXACTLY as:\nOPTIMIZED:\n<markdown>\nSUGGESTIONS:\n- <item>\n..."
  )
  user_prompt = (
    f"Target keywords (must all appear naturally): {kw_line}\n"
    f"Open-data evidence:\n{evidence[:2200]}\n\n"
    f"Original:\n{content[:1500]}\n\n"
    f"Draft to polish:\n{draft[:3500]}\n\n"
    f"Prior suggestions:\n" + "\n".join(f"- {s}" for s in suggestions[:6])
  )
  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    skip_kb_direct_match=True,
    max_tokens=min(1100, 200 + seo_optimizer_engine.count_words(draft) * 2),
    temperature=0.5,
  )
  optimized, ai_suggestions = _parse_ai_output(raw, draft)
  if seo_optimizer_engine.count_words(optimized) < max(30, seo_optimizer_engine.count_words(draft) // 3):
    return draft, suggestions, False
  merged = list(suggestions)
  for s in ai_suggestions:
    if s not in merged:
      merged.append(s)
  return optimized, merged[:12], True


async def optimize(
  provider: ModelProvider | None,
  *,
  content: str,
  keywords: list[str] | str | None = None,
  tone: str | None = None,
  language: str | None = None,
  category: str | None = None,
  use_ai: bool = True,
  use_rag: bool = True,
  variation_seed: int | None = None,
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
  content, pasted_kws = normalize_pasted_optimizer_content(content)
  if pasted_kws and not kws:
    kws = pasted_kws
  if variation_seed is None:
    variation_seed = int(time.time() * 1000) % 2_000_000_000

  ai_used = False
  rag_result: dict[str, Any] | None = None

  if is_optimizer_instruction_content(content):
    raise ValueError(
      "That text is an SEO optimizer instruction prompt, not an article to optimize. "
      "Paste your blog post or page content (e.g. a Flutter guide, product page). "
      "The tool will analyze and rewrite it automatically."
    )

  try:
    rag_result = await run_optimizer_rag_pipeline(
      content,
      keywords=kws,
      category=cat,
      tone=tone_str,
      variation_seed=variation_seed,
      use_rag=use_rag,
    )
  except ValueError:
    raise
  except Exception:
    logger.exception("SEO optimizer RAG pipeline failed; falling back to legacy path")
    rag_result = None

  if rag_result:
    optimized = rag_result["optimized_content"]
    suggestions = list(rag_result["suggestions"])
    original_metrics = rag_result["original_metrics"]
    optimized_metrics = rag_result["optimized_metrics"]
    seo_before = rag_result["seo_score_before"]
    seo_after = rag_result["seo_score_after"]
    issues_before = rag_result["issues_before"]
    issues_after = rag_result["issues_after"]
    evidence = str(rag_result.get("pipeline", {}).get("retrieval", {}))
  else:
    original_metrics = seo_optimizer_engine.content_metrics(content)
    issues_before = seo_optimizer_engine.analyze_issues(content, kws)
    seo_before = seo_optimizer_engine.seo_score_from_analysis(original_metrics, issues_before)
    optimized = content
    suggestions = [i["message"] for i in issues_before]
    if kws:
      optimized, kw_notes = rewrite_content_for_keywords(
        content, kws, topic=kws[0], seed=variation_seed,
      )
      suggestions = kw_notes + suggestions
    optimized_metrics = seo_optimizer_engine.content_metrics(optimized)
    issues_after = seo_optimizer_engine.analyze_issues(optimized, kws)
    seo_after = seo_optimizer_engine.seo_score_from_analysis(optimized_metrics, issues_after)
    evidence = ""

  if use_ai and provider is not None and (rag_result or kws):
    fast_path = (
      rag_result.get("architecture", {})
      .get("stages", {})
      .get("section_generator", {})
      .get("conservative_mode", False)
      if rag_result else False
    )
    if not fast_path or kws:
      lang_line = f" Language: {language} ({lang_code})." if language else ""
      try:
        optimized, suggestions, ai_used = await _enhance_with_ai(
          provider,
          optimized,
          content=content,
          keywords=kws,
          tone=tone_str,
          category=cat,
          lang_line=lang_line,
          evidence=evidence,
          suggestions=suggestions,
        )
        optimized_metrics = seo_optimizer_engine.content_metrics(optimized)
        issues_after = seo_optimizer_engine.analyze_issues(optimized, kws)
        seo_after = seo_optimizer_engine.seo_score_from_analysis(optimized_metrics, issues_after)
      except Exception:
        pass

  result: dict[str, Any] = {
    "category": cat,
    "language": lang_code,
    "tone": tone_str,
    "original": original_metrics,
    "optimized": optimized_metrics,
    "seo_score_before": seo_before,
    "seo_score_after": seo_after,
    "improvement": seo_after - seo_before,
    "optimized_content": optimized,
    "suggestions": suggestions[:12],
    "issues_before": issues_before,
    "issues_after": issues_after,
    "keywords": kws,
    "ai": {"enabled": use_ai, "model_used": ai_used},
    "use_rag": use_rag,
    "generator_version": rag_result.get("generator_version", "seo-optimizer-rag-v5.0") if rag_result else "legacy",
    "variation_seed": rag_result.get("variation_seed") if rag_result else None,
  }

  if rag_result:
    result["architecture"] = rag_result.get("architecture", {})
    result["elapsed_ms"] = rag_result.get("elapsed_ms")
    if rag_result.get("optimization", {}).get("seo_report"):
      result["seo_report"] = rag_result["optimization"]["seo_report"]
    result["pipeline"] = rag_result["pipeline"]
    result["optimization"] = rag_result["optimization"]
    result["rag"] = {
      "enabled": True,
      "sources_used": rag_result.get("rag_sources", []),
      "confidence": rag_result.get("pipeline", {}).get("retrieval", {}).get("confidence", 0),
    }
  else:
    result["rag"] = {"enabled": False, "sources_used": [], "confidence": 0}

  result["metrics"] = {
    "original": result["original"],
    "optimized": result["optimized"],
  }
  return result
