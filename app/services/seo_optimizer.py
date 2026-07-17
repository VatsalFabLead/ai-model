"""SEO Content Optimizer — hosted consultant strategist + conditional rewrite.

Default (use_ai): understand → audit → plan → minimal editorial rewrite via hosted LLM.
Full RAG regeneration only when strategist fails or mode=pipeline|rag|legacy|full_rag.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.engine import seo_optimizer_engine
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
  if not keywords:
    return []
  if isinstance(keywords, str):
    raw = keywords.replace("\n", ",").replace("|", ",").replace(";", ",")
    parts = raw.split(",")
  else:
    parts: list[str] = []
    for k in keywords:
      piece = str(k).replace("\n", ",").replace("|", ",").replace(";", ",")
      parts.extend(piece.split(","))
  seen: set[str] = set()
  out: list[str] = []
  for p in parts:
    piece = p.strip().strip('"').strip("'")
    if piece and piece.lower() not in seen:
      seen.add(piece.lower())
      out.append(piece)
  return out[:20]


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


def _should_use_strategist(
  *,
  use_ai: bool,
  provider: ModelProvider | None,
  mode: str | None,
) -> bool:
  if not use_ai or provider is None:
    return False
  m = (mode or "").lower().strip()
  if m in ("pipeline", "rag", "legacy"):
    return False
  return True


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
  article_type: str = "",
) -> tuple[str, list[str], bool]:
  tone_guide = seo_optimizer_engine.tone_hint(tone)
  system_prompt = (
    f"You are an expert SEO editor ({tone} — {tone_guide}). "
    f"Polish the OPTIMIZED draft. Category: {category}.{lang_line} "
    f"Article type: {article_type or 'preserve original'}. "
    "Preserve original purpose, audience, and tone. Do NOT convert into a generic tutorial. "
    "Do NOT invent Best Practices / How To / Features sections unless already present. "
    "Preserve facts; improve flow, headings, and SEO. Do not invent false claims. "
    "Respond EXACTLY as:\nOPTIMIZED:\n<markdown>\nSUGGESTIONS:\n- <item>\n..."
  )
  user_prompt = (
    f"Keywords: {', '.join(keywords) or 'infer from content'}\n"
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
    temperature=0.45,
  )
  optimized, ai_suggestions = _parse_ai_output(raw, draft)
  if seo_optimizer_engine.count_words(optimized) < max(30, seo_optimizer_engine.count_words(draft) // 3):
    return draft, suggestions, False
  merged = list(suggestions)
  for s in ai_suggestions:
    if s not in merged:
      merged.append(s)
  return optimized, merged[:12], True


def _merge_strategist_into_rewrite(
  rag_result: dict[str, Any],
  strategist_result: dict[str, Any],
) -> dict[str, Any]:
  """Keep rewritten body from RAG; overlay strategist understanding/plan/metadata when stronger."""
  out = dict(rag_result)
  strat_opt = strategist_result.get("optimization") or {}
  strat_report = (strat_opt.get("seo_report") or strategist_result.get("seo_report") or {})
  rag_opt = dict(out.get("optimization") or {})
  if strat_opt.get("faqs"):
    rag_opt["faqs"] = strat_opt["faqs"]
  if strat_opt.get("metadata"):
    rag_opt["metadata"] = strat_opt["metadata"]
  if strat_opt.get("internal_links"):
    rag_opt["internal_links"] = strat_opt["internal_links"]
  if strat_report:
    # Prefer strategist audit/plan; keep rewrite metrics from RAG
    merged_report = dict(strat_report)
    rag_report = rag_opt.get("seo_report") or {}
    if isinstance(rag_report, dict) and rag_report.get("final_metrics"):
      fm = dict(merged_report.get("final_metrics") or {})
      fm.update({
        "seo_score_before": rag_result.get("seo_score_before", fm.get("seo_score_before")),
        "seo_score_after": rag_result.get("seo_score_after", fm.get("seo_score_after")),
        "rewrite_applied": True,
      })
      merged_report["final_metrics"] = fm
      if rag_report.get("faqs") and not merged_report.get("faqs"):
        merged_report["faqs"] = rag_report["faqs"]
    rag_opt["seo_report"] = merged_report
  out["optimization"] = rag_opt
  out["article_understanding"] = strategist_result.get("article_understanding")
  out["optimization_plan"] = strategist_result.get("optimization_plan")
  out["quality_validation"] = strategist_result.get("quality_validation")
  out["seo_report"] = rag_opt.get("seo_report")
  out["rewrite_applied"] = True
  out["generator_version"] = (
    f"{strategist_result.get('generator_version', 'seo-optimizer-strategist-v1')}+"
    f"{rag_result.get('generator_version', 'rag')}"
  )
  arch = dict(out.get("architecture") or {})
  stages = dict(arch.get("stages") or {})
  strat_arch = strategist_result.get("architecture") or {}
  strat_stages = strat_arch.get("stages") or {}
  if strat_stages.get("topic_resolution"):
    stages["topic_resolution"] = strat_stages["topic_resolution"]
  if strat_stages.get("article_understanding"):
    stages["article_understanding"] = strat_stages["article_understanding"]
  arch["stages"] = stages
  out["architecture"] = arch
  # Prefer strategist keyword list
  if strategist_result.get("keywords"):
    out["keywords"] = strategist_result["keywords"]
  ai = dict(strategist_result.get("ai") or {})
  ai["rewrite_pipeline"] = True
  out["ai"] = ai
  return out


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
  rewrite: bool = True,
  mode: str | None = None,
) -> dict[str, Any]:
  content = (content or "").strip()
  if not content:
    raise ValueError("content is required")
  if len(content) > _MAX_CONTENT:
    raise ValueError(f"content exceeds maximum length of {_MAX_CONTENT} characters")

  cat = seo_optimizer_engine.normalize_category(category)
  tone_str = seo_optimizer_engine.normalize_tone(tone, cat)
  lang_code = seo_optimizer_engine.bcp47(language)
  lang_label = language or "English"
  kws = _coerce_keywords(keywords)
  content, pasted_kws = normalize_pasted_optimizer_content(content)
  user_supplied_keywords = bool(kws)
  if not kws:
    kws = pasted_kws
  if variation_seed is None:
    variation_seed = int(time.time() * 1000) % 2_000_000_000

  m = (mode or "").lower().strip()
  if m in ("audit", "plan", "strategist"):
    rewrite = False
  elif m in ("rewrite", "optimize", "full") or m == "":
    rewrite = True

  if is_optimizer_instruction_content(content):
    raise ValueError(
      "That text is an SEO optimizer instruction prompt, not an article to optimize. "
      "Paste your blog post or page content (e.g. a Flutter guide, product page). "
      "The tool will analyze and rewrite it."
    )

  strategist_result: dict[str, Any] | None = None
  if _should_use_strategist(use_ai=use_ai, provider=provider, mode=mode):
    from app.engine.seo_optimizer_strategist import generate_with_optimizer_strategist

    try:
      strategist_result = await generate_with_optimizer_strategist(
        provider,  # type: ignore[arg-type]
        content=content,
        keywords=kws,
        category=cat,
        tone=tone_str,
        language=lang_label,
        apply_rewrite=rewrite,
      )
    except Exception:
      logger.exception("SEO optimizer hosted strategist failed; falling back to RAG pipeline")
      strategist_result = None

  # Preferred path: hosted strategist with conditional (minimal) rewrite
  force_full_rag = m in ("pipeline", "rag", "legacy", "full_rag")
  if strategist_result is not None and not force_full_rag:
    result = dict(strategist_result)
    result["use_rag"] = False
    result["metrics"] = {
      "original": result["original"],
      "optimized": result["optimized"],
    }
    return result

  # Full RAG only when strategist unavailable or mode forces pipeline
  guided_kws = list(kws)
  article_type = ""
  if strategist_result:
    guided_kws = list(strategist_result.get("keywords") or kws) or guided_kws
    article_type = (
      (strategist_result.get("article_understanding") or {}).get("article_type") or ""
    )

  ai_used = False
  rag_result: dict[str, Any] | None = None
  try:
    rag_result = await run_optimizer_rag_pipeline(
      content,
      keywords=guided_kws,
      category=cat,
      tone=tone_str,
      variation_seed=variation_seed,
      use_rag=use_rag,
      user_supplied_keywords=user_supplied_keywords or bool(pasted_kws) or bool(strategist_result),
    )
  except ValueError:
    raise
  except Exception:
    logger.exception("SEO optimizer RAG pipeline failed; falling back to legacy path")
    rag_result = None

  if rag_result and strategist_result:
    rag_result = _merge_strategist_into_rewrite(rag_result, strategist_result)

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
    if strategist_result and strategist_result.get("keywords"):
      kws = list(strategist_result["keywords"])
    elif not kws:
      tr = (
        (rag_result.get("architecture") or {})
        .get("stages", {})
        .get("topic_resolution")
        or {}
      )
      primary = tr.get("primary_keyword")
      secondary = tr.get("keywords") or tr.get("secondary_keywords") or []
      if primary:
        kws = [primary] + [s for s in secondary if s and s.lower() != primary.lower()][:11]
  elif strategist_result:
    # Rewrite requested but RAG failed — still return strategist audit
    result = dict(strategist_result)
    result["suggestions"] = (
      ["Rewrite pipeline unavailable; returning audit/plan only."]
      + list(result.get("suggestions") or [])
    )[:14]
    result["metrics"] = {"original": result["original"], "optimized": result["optimized"]}
    return result
  else:
    original_metrics = seo_optimizer_engine.content_metrics(content)
    issues_before = seo_optimizer_engine.analyze_issues(content, kws)
    seo_before = seo_optimizer_engine.seo_score_from_analysis(original_metrics, issues_before)
    optimized = content
    suggestions = [i["message"] for i in issues_before]
    optimized_metrics = original_metrics
    issues_after = issues_before
    seo_after = seo_before
    evidence = ""

  if use_ai and provider is not None and rag_result and rewrite:
    fast_path = (
      rag_result.get("architecture", {})
      .get("stages", {})
      .get("section_generator", {})
      .get("conservative_mode", False)
    )
    if not fast_path:
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
          article_type=article_type,
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
    "ai": {
      "enabled": use_ai,
      "model_used": ai_used or bool(strategist_result),
      "hosted": bool(strategist_result),
      "mode": "strategist_rewrite" if strategist_result else ("rewrite" if rewrite else "pipeline"),
    },
    "use_rag": use_rag,
    "generator_version": (
      rag_result.get("generator_version", "seo-optimizer-rag-v5.2")
      if rag_result
      else "legacy"
    ),
    "variation_seed": rag_result.get("variation_seed") if rag_result else None,
    "rewrite_applied": True if rag_result else False,
  }

  if rag_result:
    result["architecture"] = rag_result.get("architecture", {})
    result["elapsed_ms"] = rag_result.get("elapsed_ms")
    if rag_result.get("optimization", {}).get("seo_report"):
      result["seo_report"] = rag_result["optimization"]["seo_report"]
    result["pipeline"] = rag_result["pipeline"]
    result["optimization"] = rag_result["optimization"]
    result["article_understanding"] = rag_result.get("article_understanding")
    result["optimization_plan"] = rag_result.get("optimization_plan")
    result["quality_validation"] = rag_result.get("quality_validation")
    if rag_result.get("ai"):
      result["ai"] = rag_result["ai"]
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
