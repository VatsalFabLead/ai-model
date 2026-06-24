"""Resume Builder AI — streamlined production pipeline.

input → spell_corrector → skill_normalizer → language_validator → ner_parser
→ summary_generator → grammar_corrector → experience_rewriter → skill_classifier
→ skill_deduplicator → resume_scorer → output
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import secrets
import time
from typing import Any, Protocol

from app.engine import resume_engine as reng
from app.engine.open_data_retrieval import OpenDoc
from app.engine.resume_narrative import (
  extract_skills_from_narrative,
  is_narrative_text,
  is_minimal_experience,
  format_structured_experience,
  rewrite_achievements_narrative,
  rewrite_certifications_narrative,
  rewrite_education_narrative,
  rewrite_experience_narrative,
  enhance_projects_section,
  rewrite_summary_narrative,
  is_meta_experience_line,
  line_has_action_verb,
  build_role_experience_bullets,
)
from app.engine.resume_open_data import (
  OPEN_DATASET_TREE,
  pick_action_verb,
  retrieve_resume_context,
)
from app.engine.resume_preprocess import (
  extract_years_experience,
  format_languages_section,
  format_years_phrase,
  normalize_languages_text,
  normalize_skills_list,
  normalize_skills_text,
  seniority_from_years,
  spell_correct_payload,
  validate_language,
)
from app.engine.seo_content_domains import make_variation_seed

GENERATOR_VERSION = "resume-builder-rag-v5.4"

PIPELINE_LAYERS = [
  "input",
  "preprocess",
  "parse",
  "generate",
  "rewrite",
  "optimize",
  "output",
]

ARCHITECTURE_FLOW = [
  "input",
  "spell_corrector",
  "skill_normalizer",
  "language_validator",
  "ner_parser",
  "summary_generator",
  "grammar_corrector",
  "experience_rewriter",
  "skill_classifier",
  "skill_deduplicator",
  "resume_scorer",
  "output",
]

_LLM_TIMEOUT_SEC = 35.0

_ACTION_VERBS = (
  "Developed", "Built", "Designed", "Implemented", "Delivered", "Optimized",
  "Created", "Integrated", "Maintained", "Collaborated on",
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


def validator(payload: dict[str, Any]) -> dict[str, Any]:
  return validate_input(payload)


class ResumeLLM(Protocol):
  async def generate_summary(self, context: dict[str, Any]) -> str | None: ...
  async def rewrite_experience(self, context: dict[str, Any]) -> str | None: ...
  async def optimize_projects(self, context: dict[str, Any]) -> str | None: ...


_DEGREE_RE = re.compile(
  r"\b(B\.?\s*E\.?|B\.?\s*Tech|M\.?\s*Tech|MBA|Ph\.?\s*D|Bachelor|Master of)\b",
  re.I,
)
_DATE_RE = re.compile(
  r"\b(?:19|20)\d{2}\b|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+\d{4}|Present|Current",
  re.I,
)
_ORG_RE = re.compile(
  r"(?:at|@)\s+([A-Z][A-Za-z0-9&.,'()\- ]{2,60}(?:"
  r"Pvt\.?|Ltd\.?|LLC|Inc\.?|Technologies|Solutions|Corp\.?)?)",
  re.I,
)
_NER_SKILL_CATALOG = (
  "flutter", "dart", "python", "java", "kotlin", "firebase", "react", "angular",
  "docker", "kubernetes", "aws", "azure", "git", "github", "sql", "mysql",
  "mongodb", "rest api", "graphql", "agile", "scrum", "figma", "android", "ios",
)


def parse_ner(payload: dict[str, Any], personal: dict[str, Any]) -> dict[str, Any]:
  """NER parser — extract entities from the full profile blob."""
  blob = " ".join(
    str(payload.get(k) or "") for k in (
      "summary", "experience", "education", "skills", "projects",
      "certifications", "achievements", "languages",
    )
  )
  low = blob.lower()
  skills_found = [t.title() if " " not in t else t.upper() for t in _NER_SKILL_CATALOG if t in low]
  skills_found.extend(_parse_skills_from_text(str(payload.get("skills") or "")))
  skills_found = list(dict.fromkeys(skills_found))
  name_parts = [p.capitalize() for p in (personal.get("full_name") or "").split() if p]
  if len(name_parts) >= 2:
    personal = {**personal, "full_name": " ".join(name_parts)}

  entities: dict[str, Any] = {
    "person": personal.get("full_name"),
    "job_titles": [personal.get("job_title")],
    "emails": re.findall(r"[\w.+-]+@[\w.-]+\.\w+", blob),
    "phones": re.findall(r"\+?\d[\d\s\-()]{7,}\d", blob),
    "organizations": list(dict.fromkeys(m.strip() for m in _ORG_RE.findall(blob)))[:6],
    "skills": skills_found[:24],
    "degrees": list(dict.fromkeys(_DEGREE_RE.findall(blob)))[:4],
    "dates": list(dict.fromkeys(_DATE_RE.findall(blob)))[:8],
    "certifications": _lines(str(payload.get("certifications") or ""))[:8],
    "projects": _lines(str(payload.get("projects") or ""))[:6],
  }
  return {
    "entities": entities,
    "entity_count": sum(len(v) if isinstance(v, list) else (1 if v else 0) for v in entities.values()),
    "personal": personal,
  }


def understand_profile(
  ner: dict[str, Any],
  personal: dict[str, Any],
  profile_text: str = "",
) -> dict[str, Any]:
  """UNDERSTAND — build career context from parsed entities (not raw echo)."""
  entities = ner.get("entities") or {}
  skills = entities.get("skills") or []
  job = personal.get("job_title") or "Professional"
  low_job = job.lower()
  if any(k in low_job for k in ("flutter", "mobile", "android", "ios")):
    domain = "mobile engineering"
  elif any(k in low_job for k in ("data", "ml", "machine learning")):
    domain = "data & AI"
  elif any(k in low_job for k in ("design", "ux", "ui")):
    domain = "product design"
  else:
    domain = "software engineering"

  years = extract_years_experience(profile_text)
  seniority = seniority_from_years(years)
  years_note = f" Approximately {years} years of experience." if years is not None else ""
  return {
    "domain": domain,
    "seniority": seniority,
    "years_experience": years,
    "core_skills": skills[:10],
    "target_role": job,
    "organizations": entities.get("organizations") or [],
    "narrative": (
      f"{personal.get('full_name', 'Candidate')} is a {seniority}-level {job} "
      f"in {domain} with strengths in {', '.join(skills[:5]) or 'cross-functional delivery'}."
      f"{years_note}"
    ),
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


def validate_name(personal: dict[str, Any]) -> dict[str, Any]:
  name = personal.get("full_name") or ""
  parts = [p for p in name.split() if p]
  valid = len(parts) >= 2 and all(p.replace(".", "").replace("-", "").isalpha() for p in parts)
  normalized = " ".join(p.capitalize() for p in parts) if parts else name
  return {
    "valid": valid,
    "normalized_name": normalized,
    "word_count": len(parts),
    "issues": [] if valid else ["name_should_have_first_and_last"],
  }


def enhance_summary(summary: str, job_title: str, skills: list[str], seed: int) -> str:
  text = (summary or "").strip()
  if not text:
    return text
  if len(text.split()) < 45:
    extra = (
      f" Skilled in {', '.join(skills[:4])} with a track record of delivering "
      f"reliable {job_title} outcomes for global teams."
    )
    text = text.rstrip(".") + "." + extra
  if "ats" not in text.lower() and seed % 2 == 0:
    text = text.rstrip(".") + ". Optimized for ATS screening and recruiter readability."
  return text.strip()


def parse_skills(raw: str | None, job_title: str, seed: int) -> list[str]:
  """skill_parser — extract skills from free text or occupation defaults."""
  return _esco_skills(job_title, raw, seed)


def classify_skills(skills: list[str], job_title: str) -> dict[str, list[str]]:
  """skill_classifier — group skills into technical / tools / soft."""
  return categorize_skills(skills, job_title)


def deduplicate_skills(skills: list[str]) -> list[str]:
  seen: set[str] = set()
  out: list[str] = []
  for s in skills:
    key = s.lower().strip()
    if key and key not in seen:
      seen.add(key)
      out.append(s.strip())
  return out


def generate_metrics(experience: str, seed: int) -> tuple[str, list[str]]:
  """metric_generator — inject quantified hints where bullets lack numbers."""
  lines = _lines(experience)
  if not lines:
    return experience, []
  metrics = ("15%", "20%", "30%", "25%", "40%", "3x", "50+")
  out: list[str] = []
  for i, ln in enumerate(lines):
    ln = re.sub(r"^[\-\*•]\s*", "", ln)
    if not ln:
      continue
    if "%" not in ln and re.search(r"\d", ln) is None:
      m = _pick(list(metrics), seed + i)
      ln = f"{ln.rstrip('.')}, contributing to ~{m} improvement in key delivery metrics"
    out.append(f"- {ln}" if not ln.startswith("-") else ln)
  joined = "\n".join(out)
  return joined, out


def extract_features(
  experience: str,
  projects: str,
  skills: list[str],
  job_title: str,
) -> dict[str, Any]:
  """feature_extractor — surface technologies and capabilities for ATS routing."""
  blob = f"{experience} {projects} {job_title} {' '.join(skills)}".lower()
  catalog = (
    "flutter", "dart", "firebase", "android", "ios", "rest api", "git",
    "kotlin", "java", "sql", "docker", "aws", "agile", "ci/cd", "mvvm",
  )
  found = [t.title() if " " not in t else t.upper() for t in catalog if t in blob]
  for s in skills[:10]:
    if s.lower() not in {f.lower() for f in found}:
      found.append(s)
  return {
    "technologies": found[:16],
    "capability_count": len(found),
    "job_family": job_title,
  }


def normalize_keywords(ats_keywords: dict[str, Any]) -> dict[str, Any]:
  """keyword_normalizer — canonical casing and synonym collapse."""
  synonyms = {
    "rest apis": "REST APIs",
    "rest api": "REST APIs",
    "github": "GitHub",
    "git hub": "GitHub",
    "ci cd": "CI/CD",
    "cross functional": "Cross-Functional",
  }
  normalized: list[str] = []
  for kw in ats_keywords.get("keywords", []):
    key = kw.lower().strip()
    normalized.append(synonyms.get(key, kw.strip().title() if " " not in kw else kw.strip()))
  return {**ats_keywords, "keywords": normalized, "normalized": True}


def _parse_skills_from_text(text: str | None) -> list[str]:
  """Parse skills from comma lists — never invent skills from substring matches."""
  if not text:
    return []
  skip_headers = {
    "technical skills", "frameworks & libraries", "tools & platforms",
    "soft skills", "frameworks", "tools", "skills",
  }
  items: list[str] = []
  for part in re.split(r"[,;\n]+", text):
    p = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", part.strip())
    p = re.sub(
      r"^(?:I have experience with|I am familiar with|I use|I have)\s+",
      "",
      p,
      flags=re.I,
    ).strip()
    if not p or len(p) < 2 or len(p) > 48:
      continue
    if p.lower() in skip_headers:
      continue
    if re.search(r"\bi\s+(?:have|use|am)\b", p, re.I):
      continue
    items.append(p)
  if items:
    return normalize_skills_list(items)
  return extract_skills_from_narrative(text or "")


def _esco_skills(job_title: str, existing: str | None, seed: int) -> list[str]:
  """Return only skills explicitly provided by the user — no invented additions."""
  parsed = _parse_skills_from_text(existing)
  if parsed:
    return normalize_skills_list(parsed)
  if existing:
    items = []
    for part in re.split(r"[,;\n]+", existing):
      p = part.strip()
      if p:
        items.append(p)
    return normalize_skills_list(items)
  return []


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
    "technical": technical,
    "soft": soft,
    "tools": tools,
  }


def generate_summary_text(
  personal: dict[str, Any],
  skills: list[str],
  experience: str,
  education: str,
  *,
  seed: int,
  understanding: dict[str, Any] | None = None,
  existing: str | None = None,
) -> str:
  """Rule-based summary — always rewrites; uses existing text only as context."""
  title = personal.get("job_title") or "Professional"
  skill_text = ", ".join(skills[:6]) if skills else "industry best practices"
  verb = _pick(list(_ACTION_VERBS), seed)
  domain = (understanding or {}).get("domain", "professional services")
  narrative = (understanding or {}).get("narrative", "")
  years_hint = " with proven hands-on delivery across real projects" if experience else ""
  context = ""
  if existing and len(existing) > 20:
    context = f" Profile context: {existing[:220].rstrip()}."
  return (
    f"{verb} {title}{years_hint} specializing in {domain}, with deep expertise in {skill_text}. "
    f"Known for measurable impact, cross-functional collaboration, and clean execution "
    f"in fast-paced environments.{context} "
    f"{narrative or ''} Focused on recruiter-ready, professional presentation."
  ).strip()


def optimize_experience(
  text: str,
  job_title: str,
  seed: int,
  improve: bool = True,
  verbs: list[str] | None = None,
  skills: list[str] | None = None,
) -> tuple[str, list[str]]:
  raw = (text or "").strip()
  if not raw or is_minimal_experience(raw):
    return format_structured_experience(raw, job_title, skills or [], seed)

  lines = [ln for ln in _lines(raw) if not is_meta_experience_line(ln)]
  bullets: list[str] = []
  for i, ln in enumerate(lines[:8]):
    ln = re.sub(r"^[\-\*]\s*", "", ln)
    if not ln or is_meta_experience_line(ln):
      continue
    if not line_has_action_verb(ln):
      ln = f"Developed {ln[0].lower() + ln[1:]}" if ln else ln
    bullets.append(f"- {ln}" if not ln.startswith("-") else ln)
  if len(bullets) < 3:
    bullets = [f"- {b}" for b in build_role_experience_bullets(job_title, skills or [])]
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


def correct_grammar(text: str) -> str:
  """Light grammar/spacing cleanup — preserves paragraph and line breaks."""
  if not (text or "").strip():
    return ""
  blocks = re.split(r"\n\s*\n", text.strip())
  fixed_blocks: list[str] = []
  for block in blocks:
    lines = []
    for ln in block.split("\n"):
      t = re.sub(r"[ \t]+", " ", ln.strip())
      t = re.sub(r"\s+([,.;:!?])", r"\1", t)
      t = re.sub(r"\bi\b", "I", t)
      if t:
        lines.append(t)
    if lines:
      fixed_blocks.append("\n".join(lines))
  return "\n\n".join(fixed_blocks)


def correct_grammar_fields(fields: dict[str, str]) -> dict[str, str]:
  return {key: correct_grammar(val) for key, val in fields.items()}


def rewrite_experience(
  text: str,
  job_title: str,
  seed: int,
  improve: bool,
) -> tuple[str, list[str], bool]:
  """Rule-based experience rewrite; LLM polish happens in service layer when use_ai=true."""
  experience, bullets = optimize_experience(text, job_title, seed, improve)
  used_llm = bool(improve and (text or "").strip())
  return experience, bullets, used_llm


def enhance_bullets(experience: str, seed: int) -> tuple[str, list[str]]:
  """Normalize bullet formatting without injecting random action verbs."""
  lines = _lines(experience)
  if not lines:
    return experience, []
  out: list[str] = []
  for ln in lines:
    ln = re.sub(r"^[\-\*•]\s*", "", ln)
    if not ln or is_meta_experience_line(ln):
      continue
    out.append(f"- {ln}" if not ln.startswith("-") else ln)
  if len(out) < 3 and "\n\n" not in experience:
    out = [f"- {b.lstrip('- ')}" for b in out]
  joined = "\n".join(out)
  return joined, out


def rewrite_achievements(text: str, seed: int) -> str:
  return analyze_achievements((text or "").strip(), seed)


def format_languages(text: str) -> str:
  body, _ = format_languages_section(text)
  return body


def deduplicate_keywords(ats_keywords: dict[str, Any]) -> dict[str, Any]:
  seen: set[str] = set()
  unique: list[str] = []
  for kw in ats_keywords.get("keywords", []):
    key = kw.lower().strip()
    if key and key not in seen:
      seen.add(key)
      unique.append(kw.strip())
  return {**ats_keywords, "keywords": unique, "deduplicated_count": len(unique)}


def build_ats_keywords(
  job_title: str,
  skills: list[str],
  docs: list[OpenDoc],
  seed: int,
  *,
  profile_text: str = "",
) -> dict[str, Any]:
  kw: set[str] = set()
  for w in re.findall(r"\w+", job_title):
    if len(w) > 2:
      kw.add(w.title())
  for s in skills:
    kw.add(s)
  onet_terms = [
    "problem solving", "critical thinking", "active learning",
    "reading comprehension", "monitoring", "coordination",
    "software development", "mobile applications", "cross functional",
  ]
  for t in onet_terms:
    kw.add(t.title())
  for d in docs[:4]:
    for w in re.findall(r"\w+", d.title):
      if len(w) > 4:
        kw.add(w.title())
  low = profile_text.lower()
  for term in (
    "flutter", "dart", "firebase", "kotlin", "java", "android", "ios",
    "rest api", "git", "github", "mysql", "sqlite", "getx", "provider",
    "mvvm", "agile", "scrum", "udemy", "coursera", "google",
  ):
    if term in low:
      kw.add(term.title() if " " not in term else term.title())
  ranked = sorted(kw)
  rotate = ranked[seed % max(1, len(ranked)):] + ranked[: seed % max(1, len(ranked))]
  cleaned = [
    k for k in rotate
    if k and len(k) > 2 and not re.search(r"\bi\s+have\b", k, re.I) and k.lower() not in {"guide", "also", "the"}
  ]
  return {
    "keywords": cleaned[:24],
    "density_target": "1.5-2.5%",
    "job_title_match": job_title,
  }


def extract_ats_keywords(
  job_title: str,
  skills: list[str],
  docs: list[OpenDoc],
  seed: int,
  *,
  profile_text: str = "",
) -> dict[str, Any]:
  return build_ats_keywords(job_title, skills, docs, seed, profile_text=profile_text)


def normalize_keywords_pipeline(ats_keywords: dict[str, Any]) -> dict[str, Any]:
  """keyword_normalizer — canonical casing + deduplication."""
  return deduplicate_keywords(normalize_keywords(ats_keywords))


async def _llm_summary(
  llm: ResumeLLM | None,
  ctx: dict[str, Any],
  fallback: str,
) -> tuple[str, bool]:
  if llm is None:
    return fallback, False
  try:
    text = await llm.generate_summary(ctx)
    if text and 40 <= len(text) <= 500 and "\n##" not in text:
      return text.strip(), True
  except Exception:
    pass
  return fallback, False


async def _llm_experience(
  llm: ResumeLLM | None,
  ctx: dict[str, Any],
  fallback: str,
) -> tuple[str, bool]:
  if llm is None:
    return fallback, False
  try:
    text = await llm.rewrite_experience(ctx)
    if text and text.count("-") >= 2:
      return text.strip(), True
  except Exception:
    pass
  return fallback, False


async def _llm_projects(
  llm: ResumeLLM | None,
  ctx: dict[str, Any],
  fallback: str,
) -> tuple[str, bool]:
  if llm is None:
    return fallback, False
  try:
    text = await llm.optimize_projects(ctx)
    if text and len(text) > 30:
      return text.strip(), True
  except Exception:
    pass
  return fallback, False


def render_markdown_sections(
  data: dict[str, Any],
  personal: dict[str, Any],
  skill_groups: dict[str, list[str]],
) -> str:
  """markdown_renderer — assemble section-based markdown body."""
  skills_flat = ", ".join(
    skill_groups.get("technical", []) + skill_groups.get("tools", []) + skill_groups.get("soft", [])
  )
  parts = [
    f"# {personal.get('full_name', '')}",
    f"**{personal.get('job_title', '')}**",
  ]
  if data.get("summary"):
    parts.extend(["", "## Professional Summary", data["summary"]])
  if skills_flat:
    parts.extend(["", "## Skills", skills_flat])
  for title, key in (
    ("Work Experience", "experience"),
    ("Education", "education"),
    ("Projects", "projects"),
    ("Certifications", "certifications"),
    ("Achievements", "achievements"),
    ("Languages", "languages"),
  ):
    body = (data.get(key) or "").strip()
    if body:
      parts.extend(["", f"## {title}", body])
  return "\n".join(parts).strip()


def render_template_output(
  data: dict[str, Any],
  personal: dict[str, Any],
  *,
  template: str,
  language: str | None,
  skill_groups: dict[str, list[str]],
) -> str:
  """template_renderer — apply resume template skin to structured content."""
  skills_flat = ", ".join(
    skill_groups.get("technical", []) + skill_groups.get("tools", []) + skill_groups.get("soft", [])
  )
  payload = {**data, "personal": personal, "skills": skills_flat}
  return reng.build_resume_markdown(
    payload, summary=data.get("summary"), template=template, language=language,
  )


def render_markdown(
  data: dict[str, Any],
  personal: dict[str, Any],
  *,
  template: str,
  language: str | None,
  skill_groups: dict[str, list[str]],
) -> str:
  return render_template_output(
    data, personal, template=template, language=language, skill_groups=skill_groups,
  )


def render_template(
  data: dict[str, Any],
  personal: dict[str, Any],
  *,
  template: str,
  language: str | None,
  skill_groups: dict[str, list[str]],
) -> str:
  return render_markdown(
    data, personal, template=template, language=language, skill_groups=skill_groups,
  )


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


def build_export_plain_text(resume_markdown: str) -> str:
  """Plain text for PDF/DOCX export (strip markdown decoration)."""
  text = resume_markdown or ""
  text = re.sub(r"^[\-\*]\s*", "", text, flags=re.MULTILINE)
  text = re.sub(r"[#*_`]", "", text)
  text = re.sub(r"📧|📱|🔗|💻", "", text)
  return re.sub(r"\n{3,}", "\n\n", text).strip()


def score_resume(fields: dict[str, Any]) -> dict[str, Any]:
  base = reng.quality_report(fields)
  completeness = base["completeness_score"]
  return {
    **base,
    "completeness_score": completeness,
    "resume_ready": completeness >= 75,
  }


def generate_pdf_bytes(plain_text: str, name: str) -> tuple[bytes | None, str | None]:
  try:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for para in plain_text.split("\n\n"):
      for line in para.split("\n"):
        pdf.multi_cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"))
      pdf.ln(2)
    out = pdf.output()
    return (bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")), None
  except ImportError:
    return None, "fpdf2 not installed"
  except Exception as exc:
    return None, str(exc)


def generate_docx_bytes(plain_text: str) -> tuple[bytes | None, str | None]:
  try:
    from docx import Document

    doc = Document()
    for block in plain_text.split("\n\n"):
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
  use_ai: bool = True,
  use_rag: bool = True,
  variation_seed: int | None = None,
  llm: ResumeLLM | None = None,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  seed = effective_variation_seed(variation_seed)
  stages: dict[str, Any] = {}
  template = reng.normalize_template(template)
  llm_active = bool(use_ai and llm is not None)

  stages["input"] = {"template": template, "use_ai": use_ai}

  validation = validator(payload)
  if not validation["valid"]:
    raise ValueError("; ".join(validation["errors"]))

  corrected, spell_meta = spell_correct_payload(payload)
  stages["spell_corrector"] = spell_meta

  skills_raw = str(corrected.get("skills") or "")
  norm_skills_text, norm_skills_preview = normalize_skills_text(skills_raw)
  if norm_skills_text:
    corrected["skills"] = norm_skills_text
  stages["skill_normalizer"] = {
    "normalized_count": len(norm_skills_preview),
    "skills": norm_skills_preview[:12],
  }

  raw_langs_field = str(corrected.get("languages") or "")
  if raw_langs_field.strip():
    norm_langs_text, spoken_lang_meta = normalize_languages_text(raw_langs_field)
    corrected["languages"] = norm_langs_text
    stages["spoken_language_validator"] = spoken_lang_meta
  else:
    stages["spoken_language_validator"] = {"validated": [], "rejected": [], "count": 0}

  lang_check = validate_language(language, corrected)
  stages["language_validator"] = lang_check
  lang_code = lang_check.get("code") or reng.bcp47(language)

  job_early = _clean(corrected.get("job_title"))
  category = reng.detect_category(job_early)
  open_ctx = await retrieve_resume_context(
    job_title=job_early,
    skills=norm_skills_preview,
    language=language,
    category=category,
    seed=seed,
    use_rag=use_rag,
  )
  stages["open_word_banks"] = {
    "sources": open_ctx["sources"],
    "action_verbs": open_ctx["action_verbs"][:10],
    "esco_category": open_ctx["esco_category"],
    "kb_hits": len(open_ctx["kb_hits"]),
    "language_code": open_ctx["language_bank"]["code"],
  }
  if open_ctx["skills"]:
    stages["skill_normalizer"]["open_skill_hints"] = open_ctx["skills"][:8]

  personal = parse_personal_info(corrected)
  profile_blob = " ".join(
    str(corrected.get(k) or "") for k in (
      "summary", "experience", "education", "skills", "projects",
      "certifications", "achievements", "languages",
    )
  )
  ner = parse_ner(corrected, personal)
  personal = ner.get("personal") or personal
  understanding = understand_profile(ner, personal, profile_blob)
  understanding["open_summary_phrases"] = open_ctx.get("summary_phrases") or []
  understanding["language_labels"] = open_ctx["language_bank"].get("section_labels") or {}
  stages["ner_parser"] = {
    "entity_count": ner.get("entity_count", 0),
    "entities": ner.get("entities"),
    "understanding": understanding,
  }

  job = personal["job_title"]
  docs = open_ctx["documents"]
  rag_sources = [s for s in open_ctx["sources"] if s not in ("resume_knowledge", "esco", "onet")]

  preview_skills = normalize_skills_list(list(dict.fromkeys(
    norm_skills_preview + _parse_skills_from_text(skills_raw)
  )))

  raw_summary = str(corrected.get("summary") or "").strip()
  summary_draft = rewrite_summary_narrative(
    raw_summary,
    personal,
    preview_skills,
    understanding,
    seed,
  )
  llm_ctx = {
    "personal": personal,
    "understanding": understanding,
    "skills": preview_skills,
    "experience": corrected.get("experience"),
    "education": corrected.get("education"),
    "language": language,
    "raw_summary": raw_summary,
    "variation_seed": seed,
    "open_phrases": open_ctx.get("summary_phrases") or [],
    "action_verbs": open_ctx.get("action_verbs") or [],
  }

  raw_experience = str(corrected.get("experience") or "").strip()
  if is_minimal_experience(raw_experience):
    exp_draft, exp_bullets = format_structured_experience(
      raw_experience, job, preview_skills, seed,
    )
    used_narrative_exp = False
  elif is_narrative_text(raw_experience):
    exp_draft, exp_bullets = rewrite_experience_narrative(raw_experience, job, seed)
    used_narrative_exp = True
  else:
    exp_draft, exp_bullets = optimize_experience(
      raw_experience, job, seed, improve=True,
      verbs=open_ctx["action_verbs"], skills=preview_skills,
    )
    used_narrative_exp = False

  exp_ctx = {**llm_ctx, "draft": exp_draft, "job_title": job}
  projects_raw = str(corrected.get("projects") or "").strip()
  projects_fallback = enhance_projects_section(projects_raw, job, preview_skills, seed)
  proj_ctx = {**llm_ctx, "draft": projects_fallback, "projects": projects_raw}

  if llm_active:
    (summary, summary_llm), (experience, exp_llm), (projects_out, proj_llm) = await asyncio.gather(
      _llm_summary(llm, llm_ctx, summary_draft),
      _llm_experience(llm, exp_ctx, exp_draft),
      _llm_projects(llm, proj_ctx, projects_fallback),
    )
  else:
    summary, summary_llm = summary_draft, False
    experience, exp_llm = exp_draft, False
    projects_out, proj_llm = projects_fallback, False

  stages["summary_generator"] = {
    "llm": summary_llm,
    "word_count": len(summary.split()),
  }

  summary = correct_grammar(summary)
  experience = correct_grammar(experience)

  stages["experience_rewriter"] = {
    "llm": exp_llm,
    "bullets": len(exp_bullets),
    "narrative_input": used_narrative_exp,
  }

  projects_out = correct_grammar(projects_out)
  stages["project_enhancer"] = {"llm": proj_llm, "lines": len(_lines(projects_out))}

  raw_ach = str(corrected.get("achievements") or "").strip()
  if is_narrative_text(raw_ach):
    achievements = correct_grammar(rewrite_achievements_narrative(raw_ach, seed))
  else:
    achievements = correct_grammar(rewrite_achievements(raw_ach, seed))

  raw_edu = str(corrected.get("education") or "").strip()
  education = (
    correct_grammar(rewrite_education_narrative(raw_edu))
    if is_narrative_text(raw_edu) else correct_grammar(process_education(raw_edu))
  ) if raw_edu else ""

  raw_certs = str(corrected.get("certifications") or "").strip()
  certifications = (
    correct_grammar(rewrite_certifications_narrative(raw_certs))
    if is_narrative_text(raw_certs) else correct_grammar(process_certifications(raw_certs))
  ) if raw_certs else ""

  raw_lang = str(corrected.get("languages") or "").strip()
  if raw_lang:
    languages, lang_fmt_meta = format_languages_section(raw_lang)
    stages["languages_formatter"] = lang_fmt_meta
  else:
    languages = ""

  grammar_fields = correct_grammar_fields({
    "summary": summary,
    "experience": experience,
    "projects": projects_out,
    "education": education,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
  })
  summary = grammar_fields["summary"]
  experience = grammar_fields["experience"]
  projects_out = grammar_fields["projects"]
  education = grammar_fields["education"]
  certifications = grammar_fields["certifications"]
  achievements = grammar_fields["achievements"]
  languages = grammar_fields["languages"]
  stages["grammar_corrector"] = {"fields_corrected": list(grammar_fields.keys())}

  skills_list = normalize_skills_list(preview_skills)
  skill_groups = classify_skills(skills_list, job)
  stages["skill_classifier"] = {**skill_groups, "count": len(skills_list)}

  skills_list = deduplicate_skills(skills_list)
  skill_groups = classify_skills(skills_list, job)
  stages["skill_deduplicator"] = {"count": len(skills_list)}

  skills_display = ", ".join(skills_list)
  data: dict[str, Any] = {
    "personal": personal,
    "summary": summary,
    "education": education,
    "experience": experience,
    "projects": projects_out,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
    "skills": skills_display,
  }

  resume_md = render_template_output(
    data, personal, template=template, language=language, skill_groups=skill_groups,
  )

  structured = {
    "full_name": personal["full_name"],
    "job_title": personal["job_title"],
    "email": personal["email"],
    "phone": personal["phone"],
    "linkedin": personal.get("linkedin") or "",
    "portfolio": personal.get("portfolio") or "",
    "education": education,
    "experience": experience,
    "skills": skills_display,
    "summary": summary,
    "projects": projects_out,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
  }
  quality = score_resume(structured)
  stages["resume_scorer"] = quality

  plain_text = build_export_plain_text(resume_md)
  pdf_bytes, pdf_err = generate_pdf_bytes(plain_text, personal["full_name"])
  docx_bytes, docx_err = generate_docx_bytes(plain_text)
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

  stages["output"] = {
    "ready": quality.get("resume_ready", False),
    "template": template,
    "word_count": len(resume_md.split()),
    "export": export,
    "llm_stages": {"summary": summary_llm, "experience": exp_llm, "projects": proj_llm},
  }

  category = reng.detect_category(job)

  return {
    "generator_version": GENERATOR_VERSION,
    "variation_seed": seed,
    "template": template,
    "template_name": template,
    "language": lang_code,
    "category": category,
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
    "projects": projects_out,
    "certifications": certifications,
    "achievements": achievements,
    "languages": languages,
    "skills": skills_display,
    "skills_list": skills_list,
    "skill_groups": skill_groups,
    "experience_bullets": exp_bullets,
    "resume_markdown": resume_md,
    "resume_ai_text": resume_md,
    "word_count": len(re.findall(r"\b[\w'-]+\b", resume_md)),
    "quality": quality,
    "export": export,
    "understanding": understanding,
    "ner": ner,
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "layers": PIPELINE_LAYERS,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "pipeline": {
      "validation": validation,
      "understanding": understanding,
      "skill_groups": skill_groups,
      "retrieval": {"sources_used": rag_sources, "document_count": len(docs)},
      "llm_used": llm_active and (summary_llm or exp_llm),
    },
    "rag": {"enabled": use_rag, "sources_used": rag_sources},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "per_request_unique": True,
  }
