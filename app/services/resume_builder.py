"""Resume Builder AI — spell-correct → normalize → parse → generate → rewrite → score → output."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.engine import resume_engine
from app.engine.resume_rag_pipeline import (
  ARCHITECTURE_FLOW,
  GENERATOR_VERSION,
  OPEN_DATASET_TREE,
  PIPELINE_LAYERS,
  ResumeLLM,
  run_resume_pipeline,
  score_resume,
)
from app.services.provider_base import ModelProvider

_MAX_TOKENS = 420
_AI_TIMEOUT_SEC = 35.0


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


def _ensure_output_fields(result: dict[str, Any]) -> dict[str, Any]:
  """Guarantee quality score and pipeline metadata on every response."""
  personal = result.get("personal_info") or {
    "full_name": result.get("full_name", ""),
    "job_title": result.get("job_title", ""),
    "email": result.get("email", ""),
    "phone": result.get("phone", ""),
    "linkedin": result.get("linkedin", ""),
    "portfolio": result.get("portfolio", ""),
  }
  fields = result.get("fields") or {
    "full_name": personal.get("full_name", ""),
    "job_title": personal.get("job_title") or result.get("job_title", ""),
    "email": personal.get("email", ""),
    "phone": personal.get("phone", ""),
    "linkedin": personal.get("linkedin", ""),
    "portfolio": personal.get("portfolio", ""),
    "education": result.get("education", ""),
    "experience": result.get("experience", ""),
    "skills": result.get("skills", ""),
    "summary": result.get("summary", ""),
    "projects": result.get("projects", ""),
    "certifications": result.get("certifications", ""),
    "achievements": result.get("achievements", ""),
    "languages": result.get("languages", ""),
  }
  if not result.get("quality"):
    result["quality"] = score_resume(fields)

  result.setdefault("generator_version", GENERATOR_VERSION)
  result.setdefault("architecture", {
    "flow": ARCHITECTURE_FLOW,
    "layers": PIPELINE_LAYERS,
    "open_datasets": OPEN_DATASET_TREE,
  })
  result.setdefault("resume_ai_text", result.get("resume_markdown", ""))
  return result


class _PipelineLLM:
  """LLM adapter for in-pipeline summary / experience / project rewriting."""

  def __init__(self, provider: ModelProvider, language: str | None) -> None:
    self._provider = provider
    self._language = language

  async def _chat(self, system: str, user: str, max_tokens: int) -> str | None:
    lang = f" Write in {self._language}." if self._language else ""
    try:
      raw = await asyncio.wait_for(
        self._provider.chat(
          [{"role": "user", "content": user}],
          system_prompt=system + lang,
          use_rag=False,
          skip_intent=True,
          max_tokens=max_tokens,
          temperature=0.55,
        ),
        timeout=_AI_TIMEOUT_SEC,
      )
      return _clean(raw) or None
    except Exception:
      return None

  async def generate_summary(self, context: dict[str, Any]) -> str | None:
    personal = context.get("personal") or {}
    understanding = context.get("understanding") or {}
    skills = ", ".join((context.get("skills") or [])[:10])
    return await self._chat(
      "Expert resume writer. Write ONE professional summary paragraph (70-100 words). "
      "Start with 'Results-driven [Job Title]'. Use third-person implied (no 'I'), "
      "but NEVER mention the candidate's name. "
      "ONLY mention technologies from the Skills list — do not invent skills. "
      "No headers, no bullets, no AI clichés like 'hiring workflows'.",
      (
        f"Role: {personal.get('job_title')}\n"
        f"Years: {understanding.get('years_experience')}\n"
        f"Skills (use ONLY these): {skills}\n"
        f"Experience notes: {context.get('experience') or ''}\n"
        f"Draft hint: {context.get('raw_summary') or context.get('draft') or ''}"
      ),
      220,
    )

  async def rewrite_experience(self, context: dict[str, Any]) -> str | None:
    return await self._chat(
      "Expert resume writer. Format work experience with:\n"
      "1) Job title on first line\n"
      "2) Company name on second line (if provided in input)\n"
      "3) Dates on third line (if provided)\n"
      "4) 4-6 bullets starting with '- '\n"
      "Use realistic responsibilities for the role. "
      "Do NOT turn duration text like '5 years' into a bullet. "
      "Do NOT invent company names or technologies not in the input. "
      "Use action verbs naturally (Developed, Built, Integrated) — never 'Reduced 5 Years'.",
      (
        f"Job title: {context.get('job_title')}\n"
        f"Skills: {', '.join((context.get('skills') or [])[:8])}\n\n"
        f"{context.get('draft') or context.get('experience') or ''}"
      ),
      _MAX_TOKENS,
    )

  async def optimize_projects(self, context: dict[str, Any]) -> str | None:
    raw = await self._chat(
      "Expert resume writer. For each project:\n"
      "1) **Project Name** on its own line\n"
      "2) One natural paragraph (2-3 sentences) describing what was built, tech used, and outcomes\n"
      "Write like a human resume — no 'Integrated the project scope' template phrases. "
      "ONLY use technologies from the Skills list or project input.",
      (
        f"Role: {(context.get('personal') or {}).get('job_title')}\n"
        f"Skills: {', '.join((context.get('skills') or [])[:8])}\n\n"
        f"{context.get('draft') or context.get('projects') or ''}"
      ),
      480,
    )
    if not raw:
      return None
    bullets = _parse_lines_or_bullets(raw)
    if len(bullets) >= 1:
      return "\n".join(b if b.startswith("-") else f"- {b}" for b in bullets)
    return raw


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
  use_ai: bool = True,
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

  llm: ResumeLLM | None = _PipelineLLM(provider, language) if use_ai and provider else None

  result = await run_resume_pipeline(
    payload,
    template=tpl,
    language=language,
    use_rag=use_rag,
    use_ai=use_ai,
    variation_seed=variation_seed,
    llm=llm,
  )

  json_out = (result.get("architecture") or {}).get("stages", {}).get("output") or {}
  llm_stages = json_out.get("llm_stages") or {}

  result["ai"] = {
    "enabled": use_ai,
    "model_used": bool(use_ai and provider and any(llm_stages.values())),
    "summary_llm": bool(llm_stages.get("summary")),
    "experience_llm": bool(llm_stages.get("experience")),
    "skills_generated": not (skills or "").strip(),
    "summary_generated": True,
    "experience_enhanced": True,
  }
  result["template"] = tpl
  return _ensure_output_fields(result)
