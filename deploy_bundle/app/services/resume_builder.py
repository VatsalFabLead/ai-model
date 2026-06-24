"""Resume Builder — RAG pipeline + optional local model polish."""

from __future__ import annotations

import re
from typing import Any

from app.engine import resume_engine
from app.engine.resume_rag_pipeline import run_resume_pipeline
from app.services.provider_base import ModelProvider

_MAX_TOKENS = 420


def _clean(text: str) -> str:
  t = (text or "").strip()
  if t.startswith("```"):
    t = re.sub(r"^```(?:\w+)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
  return re.sub(r"\n{3,}", "\n\n", t).strip()


def _parse_lines_or_bullets(text: str) -> list[str]:
  lines = []
  for ln in (text or "").splitlines():
    ln = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", ln.strip())
    if ln:
      lines.append(ln)
  return lines


def supported_categories() -> list[dict[str, Any]]:
  return resume_engine.supported_categories()


def supported_templates() -> list[dict[str, str]]:
  return resume_engine.supported_templates()


def supported_languages() -> list[dict[str, str]]:
  return resume_engine.supported_languages()


def _build_resume_markdown(*args, **kwargs) -> str:
  return resume_engine.build_resume_markdown(*args, **kwargs)


async def _ai_polish_summary(
  provider: ModelProvider,
  personal: dict[str, Any],
  draft: str,
  language: str | None,
) -> str:
  lang = f" Write in {language}." if language else ""
  try:
    raw = await provider.chat(
      [{"role": "user", "content": f"Polish this resume summary:\n{draft}"}],
      system_prompt=(
        "Resume writer. Return ONE paragraph only (60-90 words), no headers, no bullets."
        f"{lang}"
      ),
      use_rag=False,
      skip_intent=True,
      max_tokens=200,
      temperature=0.6,
    )
    text = _clean(raw)
    if 40 <= len(text) <= 400 and "\n##" not in text:
      return text
  except Exception:
    pass
  return draft


async def _ai_polish_experience(
  provider: ModelProvider,
  job_title: str,
  draft: str,
  language: str | None,
) -> str:
  lang = f" Write in {language}." if language else ""
  try:
    raw = await provider.chat(
      [{"role": "user", "content": f"Job: {job_title}\n\n{draft}"}],
      system_prompt=(
        "Rewrite as 5-7 ATS-friendly resume bullets. One per line, each starts with '- '. "
        f"Use strong action verbs and metrics when reasonable.{lang}"
      ),
      use_rag=False,
      skip_intent=True,
      max_tokens=_MAX_TOKENS,
      temperature=0.55,
    )
    bullets = _parse_lines_or_bullets(_clean(raw))
    bullets = [b if b.startswith("-") else f"- {b}" for b in bullets]
    if len(bullets) >= 3:
      return "\n".join(bullets)
  except Exception:
    pass
  return draft


async def generate(
  provider: ModelProvider | None,
  *,
  full_name: str,
  job_title: str,
  email: str,
  phone: str,
  linkedin: str | None = None,
  portfolio: str | None = None,
  education: str | None = None,
  experience: str | None = None,
  skills: str | None = None,
  summary: str | None = None,
  projects: str | None = None,
  certifications: str | None = None,
  achievements: str | None = None,
  languages: str | None = None,
  template: str = "modern",
  template_name: str | None = None,
  language: str | None = None,
  use_ai: bool = False,
  improve: bool = False,
  use_rag: bool = True,
  variation_seed: int | None = None,
) -> dict[str, Any]:
  tpl = resume_engine.normalize_template(template_name or template)
  payload = {
    "full_name": full_name,
    "job_title": job_title,
    "email": email,
    "phone": phone,
    "linkedin": linkedin,
    "portfolio": portfolio,
    "education": education,
    "experience": experience,
    "skills": skills,
    "summary": summary,
    "projects": projects,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
  }

  result = await run_resume_pipeline(
    payload,
    template=tpl,
    language=language,
    improve=improve,
    use_rag=use_rag,
    variation_seed=variation_seed,
  )

  ai_meta = {
    "enabled": use_ai,
    "skills_generated": not (skills or "").strip(),
    "summary_generated": not (summary or "").strip() or improve,
    "experience_enhanced": bool((experience or "").strip()) or improve,
    "model_used": False,
  }

  if use_ai and provider is not None:
    personal = result["personal_info"]
    if improve or not (summary or "").strip():
      result["summary"] = await _ai_polish_summary(
        provider, personal, result["summary"], language,
      )
      ai_meta["summary_generated"] = True
      ai_meta["model_used"] = True
    if (experience or "").strip() or improve:
      polished = await _ai_polish_experience(
        provider, job_title, result["experience"], language,
      )
      result["experience"] = polished
      result["experience_bullets"] = [
        b if b.startswith("-") else f"- {b}" for b in polished.splitlines() if b.strip()
      ]
      ai_meta["experience_enhanced"] = True
      ai_meta["model_used"] = True

    data = {
      "personal": personal,
      "summary": result["summary"],
      "education": result["education"],
      "experience": result["experience"],
      "projects": result["projects"],
      "certifications": result["certifications"],
      "achievements": result["achievements"],
      "languages": result["languages"],
      "skills": result["skills"],
    }
    result["resume_markdown"] = resume_engine.build_resume_markdown(
      data, summary=result["summary"], template=tpl, language=language,
    )
    result["resume_ai_text"] = result["resume_markdown"]
    from app.engine.resume_rag_pipeline import build_ats_plain_text
    result["ats_resume_text"] = build_ats_plain_text(
      personal, data, result.get("ats_keywords") or {},
    )

  result["ai"] = ai_meta
  result["template"] = tpl
  return result
