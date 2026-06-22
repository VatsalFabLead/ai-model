"""Professional Cover Letter Generator.

Creates personalized cover letters from job role, company, skills/experience,
and tone. Uses your custom model backend with template fallback.
No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.provider_base import ModelProvider

_VALID_TONES = {
  "professional",
  "casual",
  "friendly",
  "formal",
  "confident",
  "enthusiastic",
  "persuasive",
  "neutral",
}

_MAX_TOKENS = 500


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "professional"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "professional"


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


def _template_cover_letter(
  job_role: str,
  company_name: str,
  skills_experience: str,
  tone: str,
) -> str:
  role = job_role.strip() or "the open position"
  company = company_name.strip() or "your organization"
  skills = skills_experience.strip() or "relevant skills and proven experience in this field."

  return f"""Dear Hiring Manager,

I am writing to express my strong interest in the {role} role at {company}. With a track record of delivering quality work and collaborating effectively across teams, I am confident I can contribute meaningfully to your goals.

{skills}

I am particularly drawn to {company} because of its reputation and the opportunity to grow while making a tangible impact. I would welcome the chance to discuss how my background aligns with your team's needs.

Thank you for your time and consideration. I look forward to hearing from you.

Sincerely,
[Your Name]"""


async def generate_cover_letter(
  provider: ModelProvider,
  *,
  job_role: str,
  company_name: str,
  skills_experience: str,
  tone: str | None = None,
  language: str | None = None,
  applicant_name: str | None = None,
) -> dict[str, Any]:
  role = (job_role or "").strip()
  company = (company_name or "").strip()
  skills = (skills_experience or "").strip()
  if not role:
    raise ValueError("job_role is required")
  if not company:
    raise ValueError("company_name is required")
  if not skills:
    raise ValueError("skills_experience is required")

  tone_str = _normalize_tone(tone)
  lang = f" Write the entire letter in {language}." if language else ""
  name_line = f" Sign off with the name: {applicant_name}." if applicant_name else " End with 'Sincerely,' and [Your Name]."

  system_prompt = (
    f"You are an expert career coach and cover letter writer. Write a {tone_str}, "
    "personalized, one-page cover letter (3–4 short paragraphs) for the job application. "
    "Structure: greeting, why you're interested, how your skills match the role (use their "
    "details — do not invent fake employers or degrees), enthusiasm for the company, and a "
    f"professional closing.{lang}{name_line} Return only the letter text, no labels or markdown."
  )
  user_prompt = (
    f"Job role: {role}\n"
    f"Company: {company}\n"
    f"Applicant skills & experience:\n{skills}"
  )

  try:
    raw = await provider.chat(
      [{"role": "user", "content": user_prompt}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=_MAX_TOKENS,
      temperature=0.65,
    )
    letter = _clean(raw)
    if len(letter) < 120:
      raise ValueError("too short")
  except Exception:
    letter = _template_cover_letter(role, company, skills, tone_str)
    if applicant_name:
      letter = letter.replace("[Your Name]", applicant_name)

  word_count = len(re.findall(r"\b[\w'-]+\b", letter))
  return {
    "job_role": role,
    "company_name": company,
    "tone": tone_str,
    "cover_letter": letter,
    "word_count": word_count,
  }
