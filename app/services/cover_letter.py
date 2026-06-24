"""Professional Cover Letter AI Generator — full RAG production pipeline."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.engine.cover_letter_rag_pipeline import (
  ARCHITECTURE_FLOW,
  CoverLetterLLM,
  GENERATOR_VERSION,
  OPEN_DATASET_TREE,
  PIPELINE_LAYERS,
  run_cover_letter_pipeline,
  score_cover_letter,
)
from app.services.provider_base import ModelProvider

_MAX_TOKENS = 360
_AI_TIMEOUT_SEC = 90.0


def _clean(text: str) -> str:
  t = (text or "").strip()
  if t.startswith("```"):
    t = re.sub(r"^```(?:\w+)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
  t = re.sub(
    r"^(sure[,!.]?\s+)?(here(?:'s| is)|certainly|of course)[^\n:]*:\s*",
    "",
    t,
    flags=re.IGNORECASE,
  )
  t = re.sub(r"^\s*(cover\s*letter|letter|subject)\s*:\s*", "", t, flags=re.IGNORECASE)
  return re.sub(r"\n{3,}", "\n\n", t).strip()


def _ensure_output_fields(result: dict[str, Any]) -> dict[str, Any]:
  letter = result.get("cover_letter") or ""
  if not result.get("quality"):
    result["quality"] = score_cover_letter(
      letter,
      result.get("ats_keywords") or {},
      result.get("experience_analysis") or {},
      result.get("role_analysis") or {},
    )
  result.setdefault("generator_version", GENERATOR_VERSION)
  result.setdefault("architecture", {
    "flow": ARCHITECTURE_FLOW,
    "layers": PIPELINE_LAYERS,
    "open_datasets": OPEN_DATASET_TREE,
  })
  return result


class _PipelineLLM:
  """Single-call LLM adapter — one generation instead of four slow round-trips."""

  def __init__(self, provider: ModelProvider, language: str | None) -> None:
    self._provider = provider
    self._language = language

  async def generate_full_letter(self, context: dict[str, Any], draft_hint: str) -> str | None:
    tone = (context.get("tone") or {}).get("tone", "professional")
    name = context.get("applicant_name")
    var_seed = int(context.get("variation_seed") or 0)
    lang = f" Write the entire letter in {self._language}." if self._language else ""
    name_hint = f" Sign off with: Sincerely, {name}" if name else " End with Sincerely, [Your Name]"
    system = (
      f"You are an expert cover letter writer. Write one complete {tone} cover letter "
      f"(greeting, 3-4 short paragraphs, professional closing).{lang}{name_hint} "
      "Use only facts from the candidate profile — do not invent employers or degrees. "
      "Write fresh, natural prose. Do not reuse boilerplate. Return only the letter text."
    )
    user = (
      f"Variation id: {var_seed} — use distinct wording from any prior draft.\n"
      f"Job role: {context.get('job_role')}\n"
      f"Company: {context.get('company_name')}\n"
      f"Skills: {', '.join((context.get('skills') or [])[:10])}\n"
      f"Candidate background:\n{context.get('skills_experience', '')[:2000]}\n\n"
      f"Structure reference only (rewrite fully, do not copy):\n{draft_hint[:1200]}"
    )
    temp = 0.72 + (var_seed % 13) / 100.0
    try:
      raw = await asyncio.wait_for(
        self._provider.chat(
          [{"role": "user", "content": user}],
          system_prompt=system,
          use_rag=False,
          skip_intent=True,
          max_tokens=_MAX_TOKENS,
          temperature=min(0.88, temp),
        ),
        timeout=_AI_TIMEOUT_SEC,
      )
      return _clean(raw) or None
    except Exception:
      return None


async def generate_cover_letter(
  provider: ModelProvider | None,
  *,
  job_role: str,
  company_name: str,
  skills_experience: str,
  tone: str | None = None,
  language: str | None = None,
  applicant_name: str | None = None,
  use_ai: bool = True,
  use_rag: bool = True,
  variation_seed: int | None = None,
) -> dict[str, Any]:
  payload = {
    "job_role": job_role,
    "company_name": company_name,
    "skills_experience": skills_experience,
    "tone": tone,
    "applicant_name": applicant_name,
  }

  llm: CoverLetterLLM | None = _PipelineLLM(provider, language) if use_ai and provider else None

  result = await run_cover_letter_pipeline(
    payload,
    language=language,
    use_ai=use_ai,
    use_rag=use_rag,
    variation_seed=variation_seed,
    llm=llm,
  )

  json_out = (result.get("architecture") or {}).get("stages", {}).get("json_output") or {}
  llm_stages = json_out.get("llm_stages") or {}
  result["ai"] = {
    "enabled": use_ai,
    "model_used": bool(use_ai and provider and llm_stages.get("full_letter")),
    "full_letter_llm": bool(llm_stages.get("full_letter")),
    "fallback_rule_based": bool(llm_stages.get("fallback_rule_based")),
  }
  return _ensure_output_fields(result)
