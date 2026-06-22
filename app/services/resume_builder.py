"""Resume Builder — advanced, multilingual, worldwide.

Full-field resume generation with dedicated training knowledge (RAG) and
your local custom model. No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import re
from typing import Any

from app.engine import resume_engine
from app.engine.resume import _skills_for
from app.services.provider_base import ModelProvider

_MAX_TOKENS = 420


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
  return re.sub(r"\n{3,}", "\n\n", t).strip()


def _parse_lines_or_bullets(text: str) -> list[str]:
  lines = []
  for ln in (text or "").splitlines():
    ln = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", ln.strip())
    if ln:
      lines.append(ln)
  return lines


def _fallback_skills(job_title: str, existing: str | None = None) -> list[str]:
  base = _skills_for(job_title, job_title)
  if existing:
    for part in re.split(r"[,;\n]+", existing):
      p = part.strip()
      if p and p not in base:
        base.append(p)
  return base[:12]


def _guidance_block(job_title: str, language: str | None) -> str:
  # Reserved for future internal use; not injected into model prompts (small models echo it).
  return ""


def supported_categories() -> list[dict[str, Any]]:
  return resume_engine.supported_categories()


def supported_templates() -> list[dict[str, str]]:
  return resume_engine.supported_templates()


def supported_languages() -> list[dict[str, str]]:
  return resume_engine.supported_languages()


def _contact_line(personal: dict[str, Any], template: str) -> str:
  use_icons = template in {"modern", "creative"}
  parts = []
  if personal.get("email"):
    parts.append(f"{'📧 ' if use_icons else ''}{personal['email']}")
  if personal.get("phone"):
    parts.append(f"{'📱 ' if use_icons else ''}{personal['phone']}")
  if personal.get("linkedin"):
    parts.append(f"{'🔗 ' if use_icons else 'LinkedIn: '}{personal['linkedin']}")
  if personal.get("portfolio"):
    parts.append(f"{'💻 ' if use_icons else 'Portfolio: '}{personal['portfolio']}")
  sep = "  |  " if template != "classic" else "  ·  "
  return sep.join(parts) if parts else ""


def _section_block(title: str, body: str, template: str) -> str:
  body = (body or "").strip()
  if not body:
    return ""
  if template == "executive":
    return f"\n\n## ▌ {title}\n\n{body}\n"
  if template == "creative":
    return f"\n\n### ✦ {title}\n\n{body}\n"
  if template == "minimal":
    return f"\n\n## {title.upper()}\n\n{body}\n"
  return f"\n---\n\n## {title}\n\n{body}\n"


def _format_skills(skills: str | list) -> str:
  if isinstance(skills, list):
    items = [s.strip() for s in skills if str(s).strip()]
  else:
    items = [s.strip() for s in re.split(r"[,;\n]+", str(skills)) if s.strip()]
  return "\n".join(f"- {s}" for s in items)


def _build_resume_markdown(
  data: dict[str, Any],
  *,
  summary: str | None = None,
  template: str = "modern",
  language: str | None = None,
) -> str:
  labels = resume_engine.section_labels(language)
  p = data.get("personal") or {}
  name = p.get("full_name") or p.get("name") or "Your Name"
  title = p.get("job_title") or "Professional"
  contact = _contact_line(p, template)

  if template == "executive":
    header = f"# {name.upper()}\n### {title}"
  elif template == "creative":
    header = f"# ✦ {name}\n**{title}**"
  else:
    header = f"# {name}\n**{title}**"
  if contact:
    header += f"\n{contact}"

  parts = [header]

  summ = summary or data.get("summary") or ""
  if summ.strip():
    parts.append(_section_block(labels["summary"], summ.strip(), template))

  skills = data.get("skills") or ""
  if skills:
    parts.append(_section_block(labels["skills"], _format_skills(skills), template))

  exp = data.get("experience") or ""
  if exp.strip():
    parts.append(_section_block(labels["experience"], exp.strip(), template))

  edu = data.get("education") or ""
  if edu.strip():
    parts.append(_section_block(labels["education"], edu.strip(), template))

  extra = [
    (labels["projects"], data.get("projects")),
    (labels["certifications"], data.get("certifications")),
    (labels["achievements"], data.get("achievements")),
    (labels["languages"], data.get("languages")),
  ]
  for sec_title, sec_body in extra:
    if sec_body and str(sec_body).strip():
      parts.append(_section_block(sec_title, str(sec_body).strip(), template))

  return "\n".join(p for p in parts if p).strip()


def _structured_fields(data: dict[str, Any], personal: dict[str, Any]) -> dict[str, str]:
  return {
    "full_name": str(personal.get("full_name") or ""),
    "job_title": str(personal.get("job_title") or ""),
    "email": str(personal.get("email") or ""),
    "phone": str(personal.get("phone") or ""),
    "linkedin": str(personal.get("linkedin") or ""),
    "portfolio": str(personal.get("portfolio") or ""),
    "education": str(data.get("education") or ""),
    "experience": str(data.get("experience") or ""),
    "skills": str(data.get("skills") or ""),
    "summary": str(data.get("summary") or ""),
    "projects": str(data.get("projects") or ""),
    "certifications": str(data.get("certifications") or ""),
    "achievements": str(data.get("achievements") or ""),
    "languages": str(data.get("languages") or ""),
  }


async def suggest_skills(
  provider: ModelProvider,
  *,
  job_title: str,
  existing_skills: str | None = None,
  language: str | None = None,
) -> dict[str, Any]:
  job = (job_title or "").strip() or "Professional"
  lang = f" Return skill names in {language}." if language else ""
  system_prompt = (
    "You are an expert worldwide career coach. Suggest 10–14 relevant professional "
    f"skills for the given job title. Return one skill per line, no numbering.{lang}"
    + _guidance_block(job, language)
  )
  user = f"Job title: {job}"
  if existing_skills:
    user += f"\nAlready listed (keep useful ones, add more): {existing_skills}"

  try:
    raw = await provider.chat(
      [{"role": "user", "content": user}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=220,
      temperature=0.5,
    )
    skills = _parse_lines_or_bullets(_clean(raw))
    if len(skills) < 3:
      skills = _fallback_skills(job, existing_skills)
  except Exception:
    skills = _fallback_skills(job, existing_skills)

  return {
    "job_title": job,
    "category": resume_engine.detect_category(job),
    "skills": skills,
    "text": ", ".join(skills),
  }


def _is_valid_summary(text: str) -> bool:
  if not text or len(text) < 40 or len(text) > 350:
    return False
  bad = ("###", "####", "Training Knowledge", "University Name", "```", "\n- ", "\n##")
  if any(b in text for b in bad):
    return False
  if text.count("\n") > 2:
    return False
  return True


def _is_valid_bullets(text: str) -> bool:
  if not text or "###" in text or "####" in text:
    return False
  lines = [ln for ln in text.splitlines() if ln.strip()]
  if len(lines) < 2:
    return False
  return all(ln.strip().startswith("-") for ln in lines)


def _fallback_summary(
  personal: dict[str, Any],
  skills: str | None,
) -> str:
  title = personal.get("job_title") or "Professional"
  skill_list = [s.strip() for s in re.split(r"[,;\n]+", skills or "") if s.strip()][:5]
  skills_text = ", ".join(skill_list) if skill_list else "modern tools and best practices"
  return (
    f"Results-driven {title} with hands-on experience delivering reliable, user-focused solutions "
    f"across real projects. Proficient in {skills_text}, with strong problem-solving, teamwork, "
    f"and attention to detail. Committed to clean execution, continuous learning, and contributing "
    f"meaningful value to global teams from day one."
  )


async def generate_summary(
  provider: ModelProvider,
  *,
  personal: dict[str, Any],
  education: str | None = None,
  experience: str | None = None,
  skills: str | None = None,
  language: str | None = None,
) -> dict[str, Any]:
  name = personal.get("full_name") or personal.get("name") or "Candidate"
  title = personal.get("job_title") or "Professional"
  lang = f" Write entirely in {language}." if language else ""
  system_prompt = (
    "Write ONE polished professional resume summary paragraph only (3-4 sentences, 60-90 words). "
    "No name heading, no markdown headers, no bullet points, no sections, no labels. "
    "Highlight strengths, experience, and value for worldwide employers. "
    f"Plain paragraph text only.{lang}"
    + _guidance_block(title, language)
  )
  user = (
    f"Name: {name}\nRole: {title}\n"
    f"Education: {education or 'Not provided'}\n"
    f"Experience: {experience or 'Not provided'}\n"
    f"Skills: {skills or 'Not provided'}"
  )
  try:
    raw = await provider.chat(
      [{"role": "user", "content": user}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=200,
      temperature=0.6,
    )
    summary = _clean(raw)
    if not _is_valid_summary(summary):
      raise ValueError("invalid summary shape")
  except Exception:
    summary = _fallback_summary(personal, skills)

  return {"summary": summary, "word_count": len(re.findall(r"\b[\w'-]+\b", summary))}


async def optimize_bullet_points(
  provider: ModelProvider,
  *,
  job_title: str,
  experience_text: str,
  language: str | None = None,
) -> dict[str, Any]:
  title = (job_title or "").strip() or "Professional"
  exp = (experience_text or "").strip()
  if not exp:
    raise ValueError("experience_text is required")

  lang = f" Write in {language}." if language else ""
  system_prompt = (
    "Rewrite work experience as powerful, aesthetic resume bullet points for worldwide "
    "employers. Use strong action verbs and impact-focused language. Add reasonable metrics "
    "only when inferable — never invent false numbers. Return 5–7 bullets, one per line, "
    f"each starting with '- '.{lang}"
    + _guidance_block(title, language)
  )
  try:
    raw = await provider.chat(
      [{"role": "user", "content": f"Job title: {title}\n\nExperience to optimize:\n{exp}"}],
      system_prompt=system_prompt,
      use_rag=False,
      skip_intent=True,
      max_tokens=_MAX_TOKENS,
      temperature=0.55,
    )
    bullets = _parse_lines_or_bullets(_clean(raw))
    bullets = [b if b.startswith("-") else f"- {b}" for b in bullets]
    text = "\n".join(bullets)
    if len(bullets) < 2 or not _is_valid_bullets(text):
      raise ValueError("invalid bullets")
  except Exception:
    lines = [ln.strip() for ln in exp.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    bullets = []
    for ln in lines:
      ln = re.sub(r"^[\-\*]\s*", "", ln)
      if ln and not ln.startswith("*"):
        bullets.append(f"- {ln}" if not ln.startswith("-") else ln)
    if len(bullets) < 2:
      bullets = [
        f"- Delivered high-impact features as {title}, improving product quality and user satisfaction.",
        "- Collaborated with cross-functional teams across time zones to ship on schedule.",
        "- Wrote clean, maintainable code following industry best practices and code reviews.",
        "- Optimized performance and resolved critical issues for better stability.",
        "- Mentored junior team members and contributed to knowledge sharing.",
      ]
    text = "\n".join(bullets)

  return {"optimized_experience": text, "bullets": bullets}


async def generate(
  provider: ModelProvider,
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
  language: str | None = None,
  use_ai: bool = True,
) -> dict[str, Any]:
  """Single entry point: skills + summary + experience + full resume in one call."""
  personal = {
    "full_name": full_name.strip(),
    "job_title": job_title.strip(),
    "email": email.strip(),
    "phone": phone.strip(),
    "linkedin": (linkedin or "").strip() or None,
    "portfolio": (portfolio or "").strip() or None,
  }
  job = personal["job_title"]
  ai_meta = {
    "enabled": use_ai,
    "skills_generated": False,
    "summary_generated": False,
    "experience_enhanced": False,
  }

  final_skills = (skills or "").strip()
  skills_list: list[str] = []
  if use_ai and not final_skills:
    sk = await suggest_skills(provider, job_title=job, language=language)
    skills_list = sk["skills"]
    final_skills = sk["text"]
    ai_meta["skills_generated"] = True
  elif final_skills:
    skills_list = [s.strip() for s in re.split(r"[,;\n]+", final_skills) if s.strip()]

  final_experience = (experience or "").strip()
  experience_bullets: list[str] = []
  if use_ai and final_experience:
    try:
      opt = await optimize_bullet_points(
        provider,
        job_title=job,
        experience_text=final_experience,
        language=language,
      )
      final_experience = opt["optimized_experience"]
      experience_bullets = opt["bullets"]
      ai_meta["experience_enhanced"] = True
    except Exception:
      pass

  final_summary = (summary or "").strip()
  if use_ai and not final_summary:
    summ_res = await generate_summary(
      provider,
      personal=personal,
      education=education,
      experience=final_experience,
      skills=final_skills,
      language=language,
    )
    final_summary = summ_res["summary"]
    ai_meta["summary_generated"] = True

  result = await generate_full_resume(
    provider,
    personal=personal,
    education=education,
    experience=final_experience,
    skills=final_skills,
    summary=final_summary,
    projects=projects,
    certifications=certifications,
    achievements=achievements,
    languages=languages,
    template=template,
    language=language,
    use_ai_summary=False,
    use_ai_enhance=False,
  )

  result["full_name"] = personal["full_name"]
  result["job_title"] = personal["job_title"]
  result["email"] = personal["email"]
  result["phone"] = personal["phone"]
  result["linkedin"] = personal.get("linkedin") or ""
  result["portfolio"] = personal.get("portfolio") or ""
  result["skills_list"] = skills_list
  result["experience_bullets"] = experience_bullets
  result["ai"] = ai_meta
  return result


async def generate_full_resume(
  provider: ModelProvider,
  *,
  personal: dict[str, Any],
  education: str | None = None,
  experience: str | None = None,
  skills: str | None = None,
  summary: str | None = None,
  projects: str | None = None,
  certifications: str | None = None,
  achievements: str | None = None,
  languages: str | None = None,
  template: str = "modern",
  language: str | None = None,
  use_ai_summary: bool = True,
  use_ai_enhance: bool = True,
) -> dict[str, Any]:
  template = resume_engine.normalize_template(template)
  job_title = personal.get("job_title") or "Professional"
  category = resume_engine.detect_category(job_title)
  lang_code = resume_engine.bcp47(language)

  data: dict[str, Any] = {
    "personal": personal,
    "education": education or "",
    "experience": experience or "",
    "skills": skills or "",
    "summary": summary or "",
    "projects": projects or "",
    "certifications": certifications or "",
    "achievements": achievements or "",
    "languages": languages or "",
  }

  final_experience = experience or ""
  if use_ai_enhance and final_experience.strip():
    try:
      opt = await optimize_bullet_points(
        provider,
        job_title=job_title,
        experience_text=final_experience,
        language=language,
      )
      final_experience = opt["optimized_experience"]
      data["experience"] = final_experience
    except Exception:
      pass

  final_summary = summary
  if use_ai_summary and not (summary or "").strip():
    summ_res = await generate_summary(
      provider,
      personal=personal,
      education=education,
      experience=final_experience,
      skills=skills,
      language=language,
    )
    final_summary = summ_res["summary"]
    data["summary"] = final_summary

  structured = _structured_fields(data, personal)
  quality = resume_engine.quality_report(structured)

  resume_md = _build_resume_markdown(
    data,
    summary=final_summary,
    template=template,
    language=language,
  )

  return {
    "template": template,
    "language": lang_code,
    "category": category,
    "resume_markdown": resume_md,
    "summary": final_summary or "",
    "word_count": len(re.findall(r"\b[\w'-]+\b", resume_md)),
    "quality": quality,
    "fields": structured,
  }
