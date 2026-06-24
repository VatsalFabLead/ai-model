"""Advanced resume/CV engine — worldwide, multilingual, category-aware.

Uses dedicated resume training knowledge (data/resume_knowledge.jsonl).
100% custom stack — no GPT, Claude, Gemini, or proprietary APIs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESUME_KB_PATH = PROJECT_ROOT / "data" / "resume_knowledge.jsonl"

_ALL_FIELDS = [
  "full_name", "job_title", "email", "phone", "linkedin", "portfolio",
  "education", "experience", "skills", "summary", "projects",
  "certifications", "achievements", "languages",
]

_CATEGORIES: dict[str, dict[str, Any]] = {
  "technology": {
    "label": "Technology & Engineering",
    "description": "Developers, DevOps, QA, data, and IT roles",
    "roles": [
      "Software Engineer", "Flutter Developer", "Full Stack Developer",
      "Data Scientist", "DevOps Engineer", "QA Engineer", "UI/UX Designer",
    ],
  },
  "business": {
    "label": "Business & Management",
    "description": "Managers, analysts, consultants, and operations",
    "roles": [
      "Project Manager", "Business Analyst", "Product Manager",
      "Operations Manager", "Management Consultant",
    ],
  },
  "creative": {
    "label": "Creative & Media",
    "description": "Design, content, marketing, and media",
    "roles": [
      "Graphic Designer", "Content Writer", "Digital Marketer",
      "Video Editor", "Social Media Manager",
    ],
  },
  "healthcare": {
    "label": "Healthcare & Medical",
    "description": "Clinical, nursing, and health administration",
    "roles": ["Registered Nurse", "Medical Doctor", "Pharmacist", "Healthcare Administrator"],
  },
  "finance": {
    "label": "Finance & Accounting",
    "description": "Banking, accounting, and financial analysis",
    "roles": ["Accountant", "Financial Analyst", "Investment Banker", "Auditor"],
  },
  "education": {
    "label": "Education & Research",
    "description": "Teachers, professors, and researchers",
    "roles": ["Teacher", "Professor", "Research Assistant", "Academic Coordinator"],
  },
  "sales": {
    "label": "Sales & Customer Success",
    "description": "Sales, support, and client-facing roles",
    "roles": ["Sales Executive", "Account Manager", "Customer Success Manager"],
  },
  "freshers": {
    "label": "Students & Freshers",
    "description": "Interns and entry-level candidates worldwide",
    "roles": ["Intern", "Graduate Trainee", "Fresher", "Entry Level Developer"],
  },
}

_TEMPLATES: dict[str, dict[str, str]] = {
  "modern": {
    "label": "Modern",
    "description": "Clean headers, emoji contact icons, strong section dividers",
  },
  "classic": {
    "label": "Classic",
    "description": "Traditional professional layout without icons",
  },
  "executive": {
    "label": "Executive",
    "description": "Bold summary-first layout for senior roles",
  },
  "minimal": {
    "label": "Minimal",
    "description": "Compact, ATS-friendly, no decorative elements",
  },
  "creative": {
    "label": "Creative",
    "description": "Expressive layout for designers and media professionals",
  },
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en",
  "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es",
  "french": "fr", "fr": "fr",
  "german": "de", "de": "de",
  "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar",
  "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh",
  "korean": "ko", "ko": "ko",
  "italian": "it", "it": "it",
  "russian": "ru", "ru": "ru",
  "bengali": "bn", "bn": "bn",
  "tamil": "ta", "ta": "ta",
  "marathi": "mr", "mr": "mr",
  "urdu": "ur", "ur": "ur",
  "vietnamese": "vi", "vi": "vi",
  "thai": "th", "th": "th",
  "dutch": "nl", "nl": "nl",
  "polish": "pl", "pl": "pl",
  "turkish": "tr", "tr": "tr",
  "indonesian": "id", "id": "id",
}

_SECTION_LABELS: dict[str, dict[str, str]] = {
  "en": {
    "summary": "Professional Summary",
    "skills": "Skills",
    "experience": "Work Experience",
    "education": "Education",
    "projects": "Projects",
    "certifications": "Certifications",
    "achievements": "Achievements",
    "languages": "Languages",
  },
  "hi": {
    "summary": "व्यावसायिक सारांश",
    "skills": "कौशल",
    "experience": "कार्य अनुभव",
    "education": "शिक्षा",
    "projects": "परियोजनाएँ",
    "certifications": "प्रमाणपत्र",
    "achievements": "उपलब्धियाँ",
    "languages": "भाषाएँ",
  },
  "es": {
    "summary": "Resumen Profesional",
    "skills": "Habilidades",
    "experience": "Experiencia Laboral",
    "education": "Educación",
    "projects": "Proyectos",
    "certifications": "Certificaciones",
    "achievements": "Logros",
    "languages": "Idiomas",
  },
  "fr": {
    "summary": "Résumé Professionnel",
    "skills": "Compétences",
    "experience": "Expérience Professionnelle",
    "education": "Formation",
    "projects": "Projets",
    "certifications": "Certifications",
    "achievements": "Réalisations",
    "languages": "Langues",
  },
  "de": {
    "summary": "Berufliches Profil",
    "skills": "Fähigkeiten",
    "experience": "Berufserfahrung",
    "education": "Ausbildung",
    "projects": "Projekte",
    "certifications": "Zertifikate",
    "achievements": "Erfolge",
    "languages": "Sprachen",
  },
  "ar": {
    "summary": "الملخص المهني",
    "skills": "المهارات",
    "experience": "الخبرة العملية",
    "education": "التعليم",
    "projects": "المشاريع",
    "certifications": "الشهادات",
    "achievements": "الإنجازات",
    "languages": "اللغات",
  },
  "pt": {
    "summary": "Resumo Profissional",
    "skills": "Habilidades",
    "experience": "Experiência Profissional",
    "education": "Educação",
    "projects": "Projetos",
    "certifications": "Certificações",
    "achievements": "Conquistas",
    "languages": "Idiomas",
  },
  "ja": {
    "summary": "職務要約",
    "skills": "スキル",
    "experience": "職務経歴",
    "education": "学歴",
    "projects": "プロジェクト",
    "certifications": "資格",
    "achievements": "実績",
    "languages": "言語",
  },
  "zh": {
    "summary": "职业概述",
    "skills": "技能",
    "experience": "工作经历",
    "education": "教育背景",
    "projects": "项目经历",
    "certifications": "证书",
    "achievements": "成就",
    "languages": "语言",
  },
}

_resume_kb: KnowledgeBase | None = None


def get_resume_kb() -> KnowledgeBase:
  global _resume_kb
  if _resume_kb is None:
    _resume_kb = load_knowledge_base(knowledge_path=RESUME_KB_PATH)
  return _resume_kb


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def section_labels(language: str | None) -> dict[str, str]:
  code = bcp47(language)
  return _SECTION_LABELS.get(code, _SECTION_LABELS["en"])


def supported_categories() -> list[dict[str, Any]]:
  return [
    {
      "id": cat_id,
      "label": cat["label"],
      "description": cat["description"],
      "sample_roles": cat["roles"],
    }
    for cat_id, cat in _CATEGORIES.items()
  ]


def supported_templates() -> list[dict[str, str]]:
  return [{"id": k, **v} for k, v in _TEMPLATES.items()]


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"},
    {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"},
    {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"},
    {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"},
    {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"},
    {"name": "Korean", "code": "ko"},
    {"name": "Italian", "code": "it"},
    {"name": "Russian", "code": "ru"},
    {"name": "Bengali", "code": "bn"},
    {"name": "Tamil", "code": "ta"},
    {"name": "Marathi", "code": "mr"},
    {"name": "Urdu", "code": "ur"},
    {"name": "Vietnamese", "code": "vi"},
    {"name": "Thai", "code": "th"},
    {"name": "Dutch", "code": "nl"},
    {"name": "Polish", "code": "pl"},
    {"name": "Turkish", "code": "tr"},
    {"name": "Indonesian", "code": "id"},
  ]


def detect_category(job_title: str) -> str:
  low = (job_title or "").lower()
  tech_kw = ("developer", "engineer", "programmer", "devops", "data", "qa", "flutter", "python")
  if any(k in low for k in tech_kw):
    return "technology"
  creative_kw = ("designer", "writer", "marketer", "editor", "creative", "media")
  if any(k in low for k in creative_kw):
    return "creative"
  health_kw = ("nurse", "doctor", "medical", "pharma", "health")
  if any(k in low for k in health_kw):
    return "healthcare"
  finance_kw = ("accountant", "finance", "banker", "auditor", "analyst")
  if any(k in low for k in finance_kw):
    return "finance"
  edu_kw = ("teacher", "professor", "tutor", "academic")
  if any(k in low for k in edu_kw):
    return "education"
  sales_kw = ("sales", "account manager", "customer success")
  if any(k in low for k in sales_kw):
    return "sales"
  biz_kw = ("manager", "consultant", "operations", "product manager")
  if any(k in low for k in biz_kw):
    return "business"
  fresher_kw = ("intern", "fresher", "trainee", "graduate", "entry")
  if any(k in low for k in fresher_kw):
    return "freshers"
  return "technology"


def get_guidance(job_title: str, language: str | None = None) -> str:
  kb = get_resume_kb()
  lang = bcp47(language)
  category = detect_category(job_title)
  queries = [
    f"{job_title} resume best practices professional",
    f"resume {category} category worldwide ATS",
    f"resume professional summary {lang} multilingual",
  ]
  chunks: list[str] = []
  for q in queries:
    answer, score = kb.search(q)
    if answer and score > 0.05 and answer not in chunks:
      chunks.append(answer)
  return "\n\n".join(chunks[:2])


def quality_report(fields: dict[str, Any]) -> dict[str, Any]:
  filled = [f for f in _ALL_FIELDS if fields.get(f) and str(fields[f]).strip()]
  missing = [f for f in _ALL_FIELDS if f not in filled]
  score = int(round(100 * len(filled) / len(_ALL_FIELDS)))
  return {
    "completeness_score": score,
    "resume_ready": score >= 75,
    "filled_fields": filled,
    "missing_fields": missing,
    "field_count": len(filled),
  }


def normalize_template(template: str) -> str:
  t = (template or "modern").strip().lower()
  return t if t in _TEMPLATES else "modern"


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


def build_resume_markdown(
  data: dict[str, Any],
  *,
  summary: str | None = None,
  template: str = "modern",
  language: str | None = None,
) -> str:
  labels = section_labels(language)
  p = data.get("personal") or {}
  name = p.get("full_name") or p.get("name") or "Your Name"
  title = p.get("job_title") or "Professional"
  contact = _contact_line(p, template)
  template = normalize_template(template)

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
  for sec_title, key in (
    (labels["projects"], "projects"),
    (labels["certifications"], "certifications"),
    (labels["achievements"], "achievements"),
    (labels["languages"], "languages"),
  ):
    sec_body = data.get(key)
    if sec_body and str(sec_body).strip():
      parts.append(_section_block(sec_title, str(sec_body).strip(), template))
  return "\n".join(p for p in parts if p).strip()
