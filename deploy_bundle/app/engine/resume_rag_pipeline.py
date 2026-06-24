"""Resume Builder — production pipeline (open datasets, AI + ATS output).

Input → Input Validator → Personal Info Parser → Summary Generator → Education Processor
→ Experience Optimizer → Project Optimizer → Certification Processor → Achievement Analyzer
→ Skill Categorizer → Language Processor → ATS Keyword Engine → Template Renderer
→ Resume Scorer → PDF/DOCX Generator → Output
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import secrets
import time
from typing import Any

from app.engine import resume_engine as reng
from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources
from app.engine.resume import _skills_for
from app.engine.seo_content_domains import make_variation_seed

GENERATOR_VERSION = "resume-builder-rag-v2.0"

ARCHITECTURE_FLOW = [
  "input",
  "input_validator",
  "personal_info_parser",
  "summary_generator",
  "education_processor",
  "experience_optimizer",
  "project_optimizer",
  "certification_processor",
  "achievement_analyzer",
  "skill_categorizer",
  "language_processor",
  "ats_keyword_engine",
  "template_renderer",
  "resume_scorer",
  "pdf_docx_generator",
  "json_output",
]

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "JSON Resume Templates": ["jsonresume", "resume_knowledge"],
  "ESCO Skills Dataset": ["esco", "resume_knowledge"],
  "O*NET Occupation Database": ["onet", "wikipedia"],
  "Kaggle Job Description Dataset": ["kaggle_jobs", "gooaq"],
  "University Dataset": ["university", "wikidata"],
  "GeoNames Dataset": ["geonames", "wikidata"],
  "Resume NER Dataset": ["resume_ner", "resume_knowledge"],
  "Sentence Transformers": ["sentence_transformers", "local_embeddings"],
  "Gemma 3 / Llama 3": ["gemma", "llama"],
  "PDF Generator": ["fpdf", "docx"],
}

_SOURCE_ROUTE = ["wikipedia", "wikidata", "gooaq"]

_ACTION_VERBS = (
  "Led", "Built", "Delivered", "Optimized", "Designed", "Implemented",
  "Automated", "Increased", "Reduced", "Collaborated", "Managed", "Developed",
)
_TECH_SKILLS = (
  "Python", "JavaScript", "SQL", "Git", "REST APIs", "Agile", "Docker",
  "AWS", "Flutter", "Dart", "React", "Communication", "Problem Solving",
)
_SOFT_SKILLS = (
  "Leadership", "Teamwork", "Communication", "Time Management",
  "Critical Thinking", "Adaptability", "Stakeholder Management",
)


def effective_variation_seed(client_seed: int | None) -> int:
  base = make_variation_seed(client_seed)
  nonce = secrets.randbits(31) ^ (time.time_ns() & 0x7FFFFFFF)
  return make_variation_seed(base ^ nonce)


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _pick(pool: list[str], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _lines(text: str | None) -> list[str]:
  out: list[str] = []
  for ln in (text or "").splitlines():
    ln = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", ln.strip())
    if ln:
      out.append(ln)
  return out


def validate_input(payload: dict[str, Any]) -> dict[str, Any]:
  errors: list[str] = []
  for req in ("full_name", "job_title", "email", "phone"):
    if not _clean(str(payload.get(req, ""))):
      errors.append(f"missing_{req}")
  email = _clean(payload.get("email"))
  if email and "@" not in email:
    errors.append("invalid_email")
  return {
    "valid": not errors,
    "errors": errors,
    "field_count": sum(1 for k in reng._ALL_FIELDS if _clean(str(payload.get(k, "")))),
  }


def parse_personal_info(payload: dict[str, Any]) -> dict[str, Any]:
  return {
    "full_name": _clean(payload.get("full_name")),
    "job_title": _clean(payload.get("job_title")),
    "email": _clean(payload.get("email")),
    "phone": _clean(payload.get("phone")),
    "linkedin": _clean(payload.get("linkedin")) or None,
    "portfolio": _clean(payload.get("portfolio")) or None,
  }


def _esco_skills(job_title: str, existing: str | None, seed: int) -> list[str]:
  base = _skills_for(job_title, job_title)
  if existing:
    for part in re.split(r"[,;\n]+", existing):
      p = part.strip()
      if p and p not in base:
        base.append(p)
  extra = list(_TECH_SKILLS) + list(_SOFT_SKILLS)
  for i in range(4):
    s = _pick(extra, seed + i)
    if s not in base:
      base.append(s)
  return base[:14]


def categorize_skills(skills: list[str], job_title: str) -> dict[str, list[str]]:
  low_job = job_title.lower()
  technical: list[str] = []
  soft: list[str] = []
  tools: list[str] = []
  for s in skills:
    sl = s.lower()
    if any(t in sl for t in ("python", "java", "flutter", "react", "sql", "api", "docker", "aws", "git")):
      technical.append(s)
    elif any(t in sl for t in ("leadership", "communication", "team", "management", "problem")):
      soft.append(s)
    else:
      tools.append(s)
  if not technical and ("developer" in low_job or "engineer" in low_job):
    technical = skills[:6]
    tools = skills[6:]
  return {
    "technical": technical or skills[:5],
    "soft": soft or list(_SOFT_SKILLS)[:4],
    "tools": tools or skills[5:9],
  }


def generate_summary_text(
  personal: dict[str, Any],
  skills: list[str],
  experience: str,
  education: str,
  *,
  seed: int,
  improve: bool,
  existing: str | None,
) -> str:
  if existing and not improve and len(existing) > 40:
    return existing.strip()
  title = personal.get("job_title") or "Professional"
  name = personal.get("full_name") or "Candidate"
  skill_text = ", ".join(skills[:6]) if skills else "industry best practices"
  verb = _pick(list(_ACTION_VERBS), seed)
  years_hint = ""
  if experience:
    years_hint = " with proven hands-on delivery across real projects"
  return (
    f"{verb} {title}{years_hint}, combining expertise in {skill_text}. "
    f"Known for measurable impact, cross-functional collaboration, and clean execution "
    f"in fast-paced environments. {name.split()[0] if name else 'Candidate'} brings strong "
    f"problem-solving, attention to detail, and commitment to continuous improvement "
    f"for worldwide employers and ATS-friendly hiring workflows."
  )


def optimize_experience(text: str, job_title: str, seed: int, improve: bool) -> tuple[str, list[str]]:
  raw = (text or "").strip()
  if not raw:
    bullets = [
      f"- {_pick(list(_ACTION_VERBS), seed)} key initiatives as {job_title}, improving quality and delivery speed.",
      "- Collaborated with cross-functional teams to ship features on schedule.",
      "- Applied best practices for maintainable code, documentation, and code reviews.",
      "- Resolved production issues and optimized performance for better user outcomes.",
    ]
    return "\n".join(bullets), bullets

  lines = _lines(raw)
  if not improve and all(ln.startswith("-") for ln in lines if ln):
    bullets = [ln if ln.startswith("-") else f"- {ln}" for ln in lines]
    return "\n".join(bullets), bullets

  bullets: list[str] = []
  for i, ln in enumerate(lines[:8]):
    ln = re.sub(r"^[\-\*]\s*", "", ln)
    if not ln:
      continue
    verb = _pick(list(_ACTION_VERBS), seed + i)
    if not re.match(r"^[A-Z]", ln):
      ln = f"{verb} {ln[0].lower() + ln[1:]}" if ln else ln
    if not ln[0].isupper():
      ln = f"{verb} {ln}"
    bullets.append(f"- {ln}" if not ln.startswith("-") else ln)
  if len(bullets) < 3:
    bullets.append(f"- {_pick(list(_ACTION_VERBS), seed + 9)} scalable solutions aligned with business goals.")
  return "\n".join(bullets), bullets


def optimize_projects(text: str, seed: int) -> str:
  lines = _lines(text)
  if not lines:
    return ""
  out: list[str] = []
  for i, ln in enumerate(lines[:6]):
    if not ln.startswith("-"):
      ln = f"- {_pick(list(_ACTION_VERBS), seed + i)} {ln}"
    out.append(ln if ln.startswith("-") else f"- {ln}")
  return "\n".join(out)


def process_education(text: str) -> str:
  lines = _lines(text)
  if not lines:
    return ""
  return "\n".join(f"- {ln}" if not ln.startswith("-") else ln for ln in lines)


def process_certifications(text: str) -> str:
  items = [p.strip() for p in re.split(r"[,;\n]+", text or "") if p.strip()]
  if not items:
    return ""
  return "\n".join(f"- {it}" for it in items[:8])


def analyze_achievements(text: str, seed: int) -> str:
  lines = _lines(text)
  if not lines:
    return ""
  out = []
  for i, ln in enumerate(lines[:6]):
    if "%" not in ln and re.search(r"\d", ln) is None and seed % 3 == 0:
      ln = f"{ln} (measurable impact)"
    out.append(f"- {ln}" if not ln.startswith("-") else ln)
  return "\n".join(out)


def process_languages(text: str) -> str:
  items = [p.strip() for p in re.split(r"[,;\n]+", text or "") if p.strip()]
  if not items:
    return ""
  return "\n".join(f"- {lang}" for lang in items)


def build_ats_keywords(job_title: str, skills: list[str], docs: list[OpenDoc], seed: int) -> dict[str, Any]:
  kw: set[str] = set()
  for w in re.findall(r"\w+", job_title):
    if len(w) > 2:
      kw.add(w.title())
  for s in skills:
    kw.add(s)
  onet_terms = [
    "problem solving", "critical thinking", "active learning",
    "reading comprehension", "monitoring", "coordination",
  ]
  for t in onet_terms:
    kw.add(t.title())
  for d in docs[:4]:
    for w in re.findall(r"\w+", d.title):
      if len(w) > 4:
        kw.add(w.title())
  ranked = sorted(kw)
  rotate = ranked[seed % max(1, len(ranked)):] + ranked[: seed % max(1, len(ranked))]
  return {
    "keywords": rotate[:20],
    "density_target": "1.5-2.5%",
    "job_title_match": job_title,
  }


def render_template(
  data: dict[str, Any],
  personal: dict[str, Any],
  *,
  template: str,
  language: str | None,
  skill_groups: dict[str, list[str]],
) -> str:
  skills_flat = ", ".join(
    skill_groups.get("technical", []) + skill_groups.get("tools", []) + skill_groups.get("soft", [])
  )
  payload = {**data, "personal": personal, "skills": skills_flat}
  return reng.build_resume_markdown(payload, summary=data.get("summary"), template=template, language=language)


def build_ats_plain_text(
  personal: dict[str, Any],
  data: dict[str, Any],
  ats_keywords: dict[str, Any],
) -> str:
  """ATS-friendly plain text (no markdown symbols, single-column)."""
  lines = [
    personal.get("full_name", "").upper(),
    personal.get("job_title", ""),
    " | ".join(
      x for x in [
        personal.get("email"), personal.get("phone"),
        personal.get("linkedin"), personal.get("portfolio"),
      ] if x
    ),
    "",
    "PROFESSIONAL SUMMARY",
    data.get("summary", ""),
    "",
    "SKILLS",
    ", ".join(ats_keywords.get("keywords", [])),
    "",
  ]
  for title, key in (
    ("WORK EXPERIENCE", "experience"),
    ("EDUCATION", "education"),
    ("PROJECTS", "projects"),
    ("CERTIFICATIONS", "certifications"),
    ("ACHIEVEMENTS", "achievements"),
    ("LANGUAGES", "languages"),
  ):
    body = (data.get(key) or "").strip()
    if body:
      plain = re.sub(r"^[\-\*]\s*", "", body, flags=re.MULTILINE)
      plain = re.sub(r"[#*_`]", "", plain)
      lines.extend(["", title, plain])
  return "\n".join(lines).strip()


def score_resume(fields: dict[str, Any], ats_keywords: dict[str, Any]) -> dict[str, Any]:
  base = reng.quality_report(fields)
  ats_bonus = min(15, len(ats_keywords.get("keywords", [])) // 2)
  score = min(100, base["completeness_score"] + ats_bonus)
  return {
    **base,
    "completeness_score": score,
    "resume_ready": score >= 75,
    "ats_score": min(100, 50 + len(ats_keywords.get("keywords", [])) * 2),
  }


def generate_pdf_bytes(ats_text: str, name: str) -> tuple[bytes | None, str | None]:
  try:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for para in ats_text.split("\n\n"):
      for line in para.split("\n"):
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
      pdf.ln(2)
    out = pdf.output()
    return (bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")), None
  except ImportError:
    return None, "fpdf2 not installed"
  except Exception as exc:
    return None, str(exc)


def generate_docx_bytes(ats_text: str) -> tuple[bytes | None, str | None]:
  try:
    from docx import Document

    doc = Document()
    for block in ats_text.split("\n\n"):
      doc.add_paragraph(block)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue(), None
  except ImportError:
    return None, "python-docx not installed"
  except Exception as exc:
    return None, str(exc)


async def run_resume_pipeline(
  payload: dict[str, Any],
  *,
  template: str = "modern",
  language: str | None = None,
  improve: bool = False,
  use_rag: bool = True,
  variation_seed: int | None = None,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  seed = effective_variation_seed(variation_seed)
  stages: dict[str, Any] = {}
  template = reng.normalize_template(template)

  stages["input"] = {"template": template, "improve": improve}

  validation = validate_input(payload)
  stages["input_validator"] = validation
  if not validation["valid"]:
    raise ValueError("; ".join(validation["errors"]))

  personal = parse_personal_info(payload)
  stages["personal_info_parser"] = personal

  job = personal["job_title"]
  docs: list[OpenDoc] = []
  rag_sources: list[str] = []
  if use_rag:
    try:
      docs = await asyncio.wait_for(
        retrieve_from_sources(
          f"{job} resume skills occupation",
          [job.split()[0]] if job.split() else [job],
          _SOURCE_ROUTE,
          per_source=1,
          seed=seed,
        ),
        timeout=4.0,
      )
      rag_sources = sorted({d.source for d in docs})
    except asyncio.TimeoutError:
      docs = []

  skills_list = _esco_skills(job, payload.get("skills"), seed)
  skill_groups = categorize_skills(skills_list, job)
  stages["skill_categorizer"] = skill_groups

  summary = generate_summary_text(
    personal, skills_list,
    str(payload.get("experience") or ""),
    str(payload.get("education") or ""),
    seed=seed, improve=improve, existing=payload.get("summary"),
  )
  stages["summary_generator"] = {"generated": True, "word_count": len(summary.split())}

  education = process_education(str(payload.get("education") or ""))
  stages["education_processor"] = {"lines": len(_lines(education))}

  experience, exp_bullets = optimize_experience(
    str(payload.get("experience") or ""), job, seed, improve,
  )
  stages["experience_optimizer"] = {"bullets": len(exp_bullets)}

  projects = optimize_projects(str(payload.get("projects") or ""), seed)
  stages["project_optimizer"] = {"lines": len(_lines(projects))}

  certifications = process_certifications(str(payload.get("certifications") or ""))
  stages["certification_processor"] = {"items": certifications.count("-") if certifications else 0}

  achievements = analyze_achievements(str(payload.get("achievements") or ""), seed)
  stages["achievement_analyzer"] = {"lines": len(_lines(achievements))}

  languages = process_languages(str(payload.get("languages") or ""))
  stages["language_processor"] = {"items": languages.count("-") if languages else 0}

  ats_kw = build_ats_keywords(job, skills_list, docs, seed)
  stages["ats_keyword_engine"] = ats_kw

  data: dict[str, Any] = {
    "personal": personal,
    "summary": summary,
    "education": education,
    "experience": experience,
    "projects": projects,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
    "skills": ", ".join(skills_list),
  }

  resume_md = render_template(data, personal, template=template, language=language, skill_groups=skill_groups)
  stages["template_renderer"] = {"template": template, "word_count": len(resume_md.split())}

  ats_text = build_ats_plain_text(personal, data, ats_kw)

  structured = {
    "full_name": personal["full_name"],
    "job_title": personal["job_title"],
    "email": personal["email"],
    "phone": personal["phone"],
    "linkedin": personal.get("linkedin") or "",
    "portfolio": personal.get("portfolio") or "",
    "education": education,
    "experience": experience,
    "skills": ", ".join(skills_list),
    "summary": summary,
    "projects": projects,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
  }
  quality = score_resume(structured, ats_kw)
  stages["resume_scorer"] = quality

  pdf_bytes, pdf_err = generate_pdf_bytes(ats_text, personal["full_name"])
  docx_bytes, docx_err = generate_docx_bytes(ats_text)
  export: dict[str, Any] = {
    "pdf_available": pdf_bytes is not None,
    "docx_available": docx_bytes is not None,
    "pdf_error": pdf_err,
    "docx_error": docx_err,
  }
  if pdf_bytes:
    export["pdf_base64"] = base64.b64encode(pdf_bytes).decode("ascii")
  if docx_bytes:
    export["docx_base64"] = base64.b64encode(docx_bytes).decode("ascii")
  stages["pdf_docx_generator"] = export
  stages["json_output"] = {"ready": quality.get("resume_ready", False)}

  category = reng.detect_category(job)
  lang_code = reng.bcp47(language)

  return {
    "generator_version": GENERATOR_VERSION,
    "variation_seed": seed,
    "template": template,
    "template_name": template,
    "language": lang_code,
    "category": category,
    "improve": improve,
    "personal_info": personal,
    "fields": structured,
    "full_name": personal["full_name"],
    "job_title": personal["job_title"],
    "email": personal["email"],
    "phone": personal["phone"],
    "linkedin": personal.get("linkedin") or "",
    "portfolio": personal.get("portfolio") or "",
    "summary": summary,
    "education": education,
    "experience": experience,
    "projects": projects,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
    "skills": ", ".join(skills_list),
    "skills_list": skills_list,
    "skill_groups": skill_groups,
    "experience_bullets": exp_bullets,
    "resume_markdown": resume_md,
    "resume_ai_text": resume_md,
    "ats_resume_text": ats_text,
    "ats_keywords": ats_kw,
    "word_count": len(re.findall(r"\b[\w'-]+\b", resume_md)),
    "quality": quality,
    "export": export,
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "pipeline": {
      "validation": validation,
      "skill_groups": skill_groups,
      "ats_keywords": ats_kw,
      "retrieval": {"sources_used": rag_sources, "document_count": len(docs)},
    },
    "rag": {"enabled": use_rag, "sources_used": rag_sources},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "per_request_unique": True,
  }
