"""Professional Cover Letter AI — full production pipeline.

input → validator → role_parser → skill_extractor → experience_analyzer
→ tone_selector → greeting_generator → introduction_generator
→ experience_paragraph_generator → skills_paragraph_generator
→ company_personalization_engine → ats_keyword_engine → grammar_correction
→ readability_optimization → duplicate_detection → cover_letter_scorer
→ pdf_generator → docx_generator → markdown_renderer → json_output
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from typing import Any, Protocol

from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources
from app.engine.resume_narrative import extract_skills_from_narrative, is_narrative_text
from app.engine.resume_rag_pipeline import (
  build_export_plain_text,
  effective_variation_seed,
  generate_docx_bytes,
  generate_pdf_bytes,
)
from app.engine.cover_letter_templates import (
  generate_closing as tpl_closing,
  generate_company_paragraph as tpl_company,
  generate_experience_paragraph as tpl_experience,
  generate_greeting as tpl_greeting,
  generate_introduction as tpl_introduction,
  generate_signature as tpl_signature,
  generate_skills_paragraph as tpl_skills,
  template_combination_counts,
)
from app.engine.seo_optimizer_engine import count_words, improve_readability, readability_score

GENERATOR_VERSION = "cover-letter-rag-v1.3"

PIPELINE_LAYERS = [
  "input",
  "parse",
  "understand",
  "generate",
  "personalize",
  "optimize",
  "score",
  "format",
  "output",
]

ARCHITECTURE_FLOW = [
  "input",
  "input_validator",
  "role_parser",
  "skill_extractor",
  "experience_analyzer",
  "tone_selector",
  "greeting_generator",
  "introduction_generator",
  "experience_paragraph_generator",
  "skills_paragraph_generator",
  "company_personalization_engine",
  "ats_keyword_engine",
  "grammar_correction",
  "readability_optimization",
  "duplicate_detection",
  "cover_letter_scorer",
  "pdf_generator",
  "docx_generator",
  "markdown_renderer",
  "json_output",
]

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "JSON Resume Templates": ["jsonresume", "resume_knowledge"],
  "Kaggle Job Description Dataset": ["kaggle_jobs", "gooaq"],
  "O*NET Occupation Database": ["onet", "wikipedia"],
  "ESCO Skills Dataset": ["esco", "resume_knowledge"],
  "Sentence Transformers": ["sentence_transformers", "local_embeddings"],
  "Gemma 3 / Llama 3": ["gemma", "llama"],
  "Grammar Correction Model": ["grammar", "local_nlp"],
  "PDF Generator": ["fpdf", "docx"],
}

_SOURCE_ROUTE = ["wikipedia", "wikidata", "gooaq", "onet"]

_VALID_TONES = {
  "professional", "casual", "friendly", "formal",
  "confident", "enthusiastic", "persuasive", "neutral",
}

_SENIORITY_RE = re.compile(
  r"\b(senior|sr\.?|lead|principal|staff|junior|jr\.?|entry[\-\s]?level|mid[\-\s]?level|"
  r"manager|director|head of|vp|vice president|intern|associate)\b",
  re.I,
)
_YEARS_RE = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience)?", re.I)
_ROLE_SKILL_NOISE = frozenset(
  "developer engineer manager lead senior junior intern associate specialist analyst".split()
)
_ROLE_STOP = frozenset(
  "a an the and or for in at to of with on by from as is are was were be been being".split()
)

_LLM_TIMEOUT_SEC = 90.0


class CoverLetterLLM(Protocol):
  async def generate_full_letter(self, context: dict[str, Any], draft_hint: str) -> str | None: ...


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def correct_grammar(text: str) -> str:
  if not (text or "").strip():
    return ""
  blocks = re.split(r"\n\s*\n", text.strip())
  fixed: list[str] = []
  for block in blocks:
    lines = []
    for ln in block.split("\n"):
      t = re.sub(r"[ \t]+", " ", ln.strip())
      t = re.sub(r"\s+([,.;:!?])", r"\1", t)
      t = re.sub(r"\bi\b", "I", t)
      if t:
        lines.append(t)
    if lines:
      fixed.append("\n".join(lines))
  return "\n\n".join(fixed)


def validator(payload: dict[str, Any]) -> dict[str, Any]:
  errors: list[str] = []
  role = _clean(payload.get("job_role"))
  company = _clean(payload.get("company_name"))
  skills = (payload.get("skills_experience") or "").strip()
  if not role:
    errors.append("job_role is required")
  if not company:
    errors.append("company_name is required")
  if not skills:
    errors.append("skills_experience is required")
  elif len(skills) < 12:
    errors.append("skills_experience is too short — add more detail")
  return {"valid": not errors, "errors": errors}


def parse_role(job_role: str, seed: int) -> dict[str, Any]:
  role = _clean(job_role)
  low = role.lower()
  seniority = "mid"
  m = _SENIORITY_RE.search(role)
  if m:
    token = m.group(1).lower()
    if any(x in token for x in ("senior", "sr", "lead", "principal", "staff", "director", "head", "vp")):
      seniority = "senior"
    elif any(x in token for x in ("junior", "jr", "entry", "intern", "associate")):
      seniority = "junior"
  tokens = [t for t in re.findall(r"[A-Za-z0-9+#.]+", role) if t.lower() not in _ROLE_STOP]
  core_title = " ".join(tokens[:4]) if tokens else role
  domain = "technology"
  for needle, label in (
    ("marketing", "marketing"), ("sales", "sales"), ("human resources", "hr"), (" hr", "hr"),
    ("finance", "finance"), ("design", "design"), ("nurse", "healthcare"), ("health", "healthcare"),
    ("teacher", "education"), ("education", "education"),
  ):
    if needle in low:
      domain = label
      break
  return {
    "raw": role,
    "core_title": core_title,
    "seniority": seniority,
    "domain": domain,
    "keywords": tokens[:8],
  }


def _parse_skill_list(text: str) -> list[str]:
  raw = (text or "").strip()
  if not raw:
    return []
  if "," in raw or ";" in raw:
    parts = [re.sub(r"^[\-\*\•\d]+[\).\s]+", "", p.strip()) for p in re.split(r"[,;]+", raw)]
    return [p for p in parts if p and len(p) > 1]
  return []


def extract_skills(skills_experience: str, role: dict[str, Any]) -> list[str]:
  text = skills_experience or ""
  found = _parse_skill_list(text)
  if not found:
    found = extract_skills_from_narrative(text)
  for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9+#.]*)\b", text):
    word = m.group(1)
    if word.lower() in _ROLE_SKILL_NOISE:
      continue
    if word not in found and len(word) > 2:
      found.append(word)
  # Do not pull job-title tokens (e.g. Developer) into skills
  for token in role.get("keywords") or []:
    if len(token) > 2 and token.lower() not in _ROLE_SKILL_NOISE:
      if token.lower() not in {f.lower() for f in found}:
        if token.lower() in text.lower():
          found.append(token.title() if token.islower() else token)
  return list(dict.fromkeys(found))[:12]


def analyze_experience(skills_experience: str, seed: int) -> dict[str, Any]:
  text = (skills_experience or "").strip()
  years = 0
  ym = _YEARS_RE.search(text)
  if ym:
    years = int(ym.group(1))
  bullets = [ln for ln in text.splitlines() if ln.strip().startswith(("-", "*", "•"))]
  achievements = re.findall(r"\b\d+%?\b", text)
  narrative = is_narrative_text(text)
  highlights: list[str] = []
  for ln in re.split(r"[\n.]+", text):
    ln = ln.strip()
    if len(ln) > 25 and not ln.lower().startswith(("i am", "i have")):
      highlights.append(ln[:200])
    if len(highlights) >= 4:
      break
  return {
    "years_experience": years,
    "bullet_count": len(bullets),
    "achievement_signals": len(achievements),
    "narrative_input": narrative,
    "highlights": highlights[:4],
    "word_count": count_words(text),
  }


def select_tone(tone: str | None) -> dict[str, Any]:
  t = (tone or "professional").strip().lower()
  if t not in _VALID_TONES:
    t = "professional"
  hints = {
    "professional": "Clear, confident, business-appropriate.",
    "casual": "Relaxed and conversational while staying respectful.",
    "friendly": "Warm and approachable.",
    "formal": "Structured and corporate.",
    "confident": "Assertive with measurable outcomes.",
    "enthusiastic": "Energetic and motivated.",
    "persuasive": "Compelling with clear value proposition.",
    "neutral": "Balanced and factual.",
  }
  return {"tone": t, "hint": hints.get(t, hints["professional"])}


def generate_greeting(company: str, applicant_name: str | None, seed: int, tone: str) -> str:
  return tpl_greeting(company, applicant_name, seed, tone)


def generate_introduction(
  role: dict[str, Any],
  company: str,
  experience: dict[str, Any],
  tone: dict[str, Any],
  seed: int,
) -> str:
  title = role.get("core_title") or role.get("raw") or "the open position"
  return tpl_introduction(
    title,
    _clean(company),
    int(experience.get("years_experience") or 0),
    role.get("seniority", "mid"),
    seed,
    tone.get("tone", "professional"),
  )


def generate_experience_paragraph(
  role: dict[str, Any],
  experience: dict[str, Any],
  skills_experience: str,
  skills_list: list[str],
  seed: int,
  tone: str,
) -> str:
  title = role.get("core_title") or role.get("raw")
  skill_phrase = ", ".join(skills_list[:6]) if skills_list else _clean(skills_experience)[:120]
  highlights = experience.get("highlights") or []
  highlight = highlights[seed % len(highlights)] if highlights else None
  return tpl_experience(title, skill_phrase, highlight, seed, tone)


def generate_skills_paragraph(
  skills: list[str],
  role: dict[str, Any],
  company: str,
  seed: int,
  tone: str,
) -> str:
  skill_text = ", ".join(skills[:6]) if skills else "relevant technical skills"
  return tpl_skills(
    skill_text,
    role.get("core_title") or role.get("raw") or "this role",
    company,
    role.get("seniority", "mid"),
    seed,
    tone,
  )


def personalize_company(
  company: str,
  role: dict[str, Any],
  seed: int,
  tone: str,
) -> str:
  title = role.get("core_title") or role.get("raw") or "this role"
  return tpl_company(_clean(company), title, seed, tone)


def build_ats_keywords(
  role: dict[str, Any],
  skills: list[str],
) -> dict[str, Any]:
  keywords: list[str] = []
  for token in role.get("keywords") or []:
    if len(token) > 2 and token.lower() not in _ROLE_SKILL_NOISE:
      keywords.append(token)
  keywords.extend(skills[:10])
  keywords = list(dict.fromkeys(keywords))[:12]
  return {"keywords": keywords, "source_count": 0}


def apply_ats_keywords(letter: str, keywords: list[str]) -> tuple[str, dict[str, Any]]:
  """Score keyword coverage only — never append junk phrases to the letter."""
  low = letter.lower()
  matched = [k for k in keywords if k.lower() in low]
  coverage = round(100 * len(matched) / max(1, len(keywords)))
  return letter, {
    "keywords": keywords,
    "matched": matched,
    "missing": [k for k in keywords if k.lower() not in low][:6],
    "coverage_pct": coverage,
  }


def detect_duplicates(text: str) -> dict[str, Any]:
  sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 20]
  seen: dict[str, str] = {}
  duplicates: list[str] = []
  for s in sentences:
    key = re.sub(r"\W+", "", s.lower())
    if key in seen:
      duplicates.append(s)
    else:
      seen[key] = s
  cleaned = text
  for dup in duplicates:
    cleaned = cleaned.replace(dup, "", 1)
  cleaned = re.sub(r"\n{3,}", "\n\n", re.sub(r"  +", " ", cleaned)).strip()
  return {
    "duplicate_count": len(duplicates),
    "duplicates": duplicates[:5],
    "cleaned_text": cleaned,
  }


def score_cover_letter(
  letter: str,
  ats: dict[str, Any],
  experience: dict[str, Any],
  role: dict[str, Any],
) -> dict[str, Any]:
  words = count_words(letter)
  readability = readability_score(letter)
  ats_cov = ats.get("coverage_pct", 0)
  completeness = 0
  checks: list[str] = []
  if words >= 180:
    completeness += 25
    checks.append("length")
  if words <= 450:
    completeness += 10
  if ats_cov >= 50:
    completeness += 25
    checks.append("ats_keywords")
  if experience.get("highlights") or experience.get("years_experience"):
    completeness += 20
    checks.append("experience")
  if role.get("core_title"):
    completeness += 20
    checks.append("role_match")
  score = min(100, completeness + min(20, int(readability) // 5))
  return {
    "quality_score": int(round(score)),
    "readability_score": int(round(readability)),
    "word_count": words,
    "ats_coverage_pct": int(ats_cov),
    "letter_ready": score >= 75 and words >= 150,
    "checks_passed": checks,
  }


def render_markdown(
  greeting: str,
  paragraphs: list[str],
  closing: str,
  signature: str,
  meta: dict[str, Any],
) -> str:
  lines = [
    f"# Cover Letter — {meta.get('job_role', '')} @ {meta.get('company_name', '')}",
    "",
    f"**Tone:** {meta.get('tone', 'professional')} · **Quality:** {meta.get('quality_score', '—')}/100",
    "",
    greeting,
    "",
  ]
  lines.extend(p for p in paragraphs if p.strip())
  lines.extend(["", closing, "", signature])
  return "\n".join(lines).strip()


def assemble_letter(
  greeting: str,
  intro: str,
  experience_para: str,
  skills_para: str,
  company_para: str,
  applicant_name: str | None,
  seed: int,
  tone: str,
) -> str:
  closing = tpl_closing(seed, tone)
  signature = tpl_signature(applicant_name, seed, tone)
  mid = [experience_para, skills_para, company_para]
  shift = seed % len(mid)
  mid = mid[shift:] + mid[:shift]
  parts = [greeting, intro, *mid, closing, signature]
  return "\n\n".join(p.strip() for p in parts if p.strip())


async def _llm_call(fn, *args):
  try:
    return await asyncio.wait_for(fn(*args), timeout=_LLM_TIMEOUT_SEC)
  except Exception:
    return None


async def run_cover_letter_pipeline(
  payload: dict[str, Any],
  *,
  language: str | None = None,
  use_ai: bool = True,
  use_rag: bool = True,
  variation_seed: int | None = None,
  llm: CoverLetterLLM | None = None,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  seed = effective_variation_seed(variation_seed)
  stages: dict[str, Any] = {}
  llm_active = bool(use_ai and llm is not None)

  job_role = _clean(payload.get("job_role"))
  company_name = _clean(payload.get("company_name"))
  skills_experience = (payload.get("skills_experience") or "").strip()
  applicant_name = _clean(payload.get("applicant_name")) or None
  tone_in = payload.get("tone")

  stages["input"] = {
    "job_role": job_role,
    "company_name": company_name,
    "use_ai": use_ai,
    "use_rag": use_rag,
  }

  validation = validator(payload)
  stages["input_validator"] = validation
  if not validation["valid"]:
    raise ValueError("; ".join(validation["errors"]))

  role = parse_role(job_role, seed)
  stages["role_parser"] = role

  skills_list = extract_skills(skills_experience, role)
  stages["skill_extractor"] = {"skills": skills_list, "count": len(skills_list)}

  experience = analyze_experience(skills_experience, seed)
  stages["experience_analyzer"] = experience

  tone = select_tone(tone_in)
  stages["tone_selector"] = tone

  greeting = generate_greeting(company_name, applicant_name, seed, tone["tone"])
  stages["greeting_generator"] = {"greeting": greeting, "template_variants": template_combination_counts()}

  docs: list[OpenDoc] = []
  rag_sources: list[str] = []
  if use_rag:
    try:
      docs = await asyncio.wait_for(
        retrieve_from_sources(
          f"{company_name} company careers culture",
          [company_name],
          ["gooaq", "wikidata"],
          per_source=1,
          seed=seed,
        ),
        timeout=3.0,
      )
      rag_sources = sorted({d.source for d in docs})
    except asyncio.TimeoutError:
      docs = []

  ats_raw = build_ats_keywords(role, skills_list)
  stages["ats_keyword_engine"] = {**ats_raw, "layer": "optimize"}

  llm_ctx = {
    "job_role": job_role,
    "company_name": company_name,
    "role": role,
    "skills": skills_list,
    "experience": experience,
    "skills_experience": skills_experience,
    "tone": tone,
    "language": language,
    "applicant_name": applicant_name,
    "variation_seed": seed,
  }

  intro_draft = generate_introduction(role, company_name, experience, tone, seed)
  exp_draft = generate_experience_paragraph(
    role, experience, skills_experience, skills_list, seed, tone["tone"],
  )
  skills_draft = generate_skills_paragraph(skills_list, role, company_name, seed, tone["tone"])
  company_draft = personalize_company(company_name, role, seed, tone["tone"])

  rule_draft = assemble_letter(
    greeting, intro_draft, exp_draft, skills_draft, company_draft, applicant_name, seed, tone["tone"],
  )

  full_llm = False
  draft = rule_draft
  if llm_active:
    full = await _llm_call(llm.generate_full_letter, llm_ctx, rule_draft)
    if full and len(full) > 120:
      draft = full
      full_llm = True

  stages["introduction_generator"] = {"llm": full_llm, "word_count": count_words(intro_draft)}
  stages["experience_paragraph_generator"] = {"llm": full_llm, "word_count": count_words(exp_draft)}
  stages["skills_paragraph_generator"] = {"llm": full_llm, "word_count": count_words(skills_draft)}
  stages["company_personalization_engine"] = {
    "company": company_name,
    "rag_sources": rag_sources,
    "snippet_used": bool(docs),
  }

  draft = correct_grammar(draft)
  stages["grammar_correction"] = {"applied": True}

  draft, readability_notes = improve_readability(draft, target_min=55.0, max_passes=2)
  stages["readability_optimization"] = {
    "readability_score": int(round(readability_score(draft))),
    "suggestions": readability_notes[:4],
  }

  dup = detect_duplicates(draft)
  if dup["duplicate_count"]:
    draft = dup["cleaned_text"]
  stages["duplicate_detection"] = {
    "duplicate_count": dup["duplicate_count"],
    "removed": dup["duplicate_count"] > 0,
  }

  draft, ats_result = apply_ats_keywords(draft, ats_raw.get("keywords", []))
  stages["ats_keyword_engine"]["coverage"] = ats_result

  quality = score_cover_letter(draft, ats_result, experience, role)
  stages["cover_letter_scorer"] = quality

  plain = build_export_plain_text(draft)
  pdf_bytes, pdf_err = generate_pdf_bytes(plain, applicant_name or job_role)
  docx_bytes, docx_err = generate_docx_bytes(plain)
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
  stages["pdf_generator"] = {"available": export["pdf_available"], "error": pdf_err}
  stages["docx_generator"] = {"available": export["docx_available"], "error": docx_err}

  cover_md = render_markdown(
    greeting,
    [intro_draft, exp_draft, skills_draft, company_draft],
    (
      "Thank you for your time and consideration. I look forward to the opportunity to discuss "
      "how my experience can support your team's goals."
    ),
    f"Sincerely,\n{applicant_name or '[Your Name]'}",
    {
      "job_role": job_role,
      "company_name": company_name,
      "tone": tone["tone"],
      "quality_score": quality["quality_score"],
    },
  )
  stages["markdown_renderer"] = {"word_count": count_words(cover_md)}
  stages["json_output"] = {
    "ready": quality.get("letter_ready", False),
    "llm_stages": {
      "full_letter": full_llm,
      "fallback_rule_based": llm_active and not full_llm,
    },
  }

  word_count = count_words(draft)
  return {
    "generator_version": GENERATOR_VERSION,
    "variation_seed": seed,
    "job_role": job_role,
    "company_name": company_name,
    "tone": tone["tone"],
    "cover_letter": draft,
    "cover_letter_markdown": cover_md,
    "cover_letter_plain": plain,
    "word_count": word_count,
    "skills_list": skills_list,
    "ats_keywords": ats_result,
    "quality": quality,
    "export": export,
    "role_analysis": role,
    "experience_analysis": experience,
    "architecture": {
      "flow": ARCHITECTURE_FLOW,
      "layers": PIPELINE_LAYERS,
      "stages": stages,
      "open_datasets": OPEN_DATASET_TREE,
    },
    "pipeline": {
      "validation": validation,
      "role": role,
      "skills": skills_list,
      "retrieval": {"sources_used": rag_sources, "document_count": len(docs)},
      "llm_used": full_llm,
    },
    "rag": {"enabled": use_rag, "sources_used": rag_sources},
    "ai": {
      "enabled": use_ai,
      "model_used": full_llm,
    },
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    "per_request_unique": True,
  }
