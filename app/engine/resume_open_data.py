"""Resume open free datasets + word banks — ESCO, O*NET, KB, live retrieval.

Sources: resume_knowledge.jsonl, Wikipedia, Wikidata, GooAQ, Stack Exchange,
ConceptNet (via open_data_retrieval routes), O*NET occupation terms, ESCO skills.
"""

from __future__ import annotations

import asyncio
import re
from functools import lru_cache
from typing import Any

from app.engine import resume_engine as reng
from app.engine.knowledge import KnowledgeBase, load_knowledge_base
from app.engine.open_data_retrieval import OpenDoc, retrieve_from_sources
from app.engine.resume_engine import RESUME_KB_PATH

OPEN_DATASET_TREE: dict[str, list[str]] = {
  "JSON Resume Templates": ["jsonresume", "resume_knowledge"],
  "ESCO Skills Dataset": ["esco", "resume_knowledge", "local_faiss"],
  "O*NET Occupation Database": ["onet", "wikipedia", "wikidata"],
  "Kaggle Job Description Dataset": ["kaggle_jobs", "gooaq"],
  "Resume NER Dataset": ["resume_ner", "resume_knowledge"],
  "Sentence Transformers": ["sentence_transformers", "local_embeddings"],
  "Gemma 3 / Llama 3": ["gemma", "llama"],
  "Grammar Correction Model": ["grammar", "local_nlp"],
  "Multilingual Word Banks": ["resume_knowledge", "wikipedia", "conceptnet"],
  "Language Open Corpora": ["wikipedia", "wikidata", "gooaq", "dolly"],
  "PDF Generator": ["fpdf", "docx"],
}

_RESUME_SOURCE_ROUTE = [
  "wikipedia",
  "wikidata",
  "gooaq",
  "stackexchange",
  "conceptnet",
  "local_faiss",
]

# O*NET-style universal skills (public occupation taxonomy)
_ONET_SOFT_SKILLS = (
  "Active Listening", "Critical Thinking", "Complex Problem Solving",
  "Reading Comprehension", "Writing", "Speaking", "Monitoring",
  "Social Perceptiveness", "Coordination", "Time Management",
  "Judgment and Decision Making", "Active Learning",
)

# ESCO-style technical skill seeds
_ESCO_TECH_SKILLS: dict[str, tuple[str, ...]] = {
  "technology": (
    "Python", "JavaScript", "SQL", "Git", "REST APIs", "Docker", "AWS",
    "Flutter", "Dart", "React", "Kotlin", "Agile", "Scrum", "CI/CD",
    "Firebase", "GraphQL", "Kubernetes", "PostgreSQL", "TypeScript",
  ),
  "business": (
    "Project Management", "Stakeholder Management", "Business Analysis",
    "Microsoft Excel", "PowerPoint", "Strategic Planning", "KPI Tracking",
  ),
  "creative": (
    "Figma", "Adobe Creative Suite", "Content Strategy", "SEO",
    "Copywriting", "Brand Design", "Canva",
  ),
  "healthcare": (
    "Patient Care", "HIPAA", "Electronic Health Records", "Clinical Documentation",
    "BLS", "Medical Terminology",
  ),
  "finance": (
    "Financial Modeling", "Excel", "GAAP", "IFRS", "SAP", "Risk Analysis",
    "Budgeting", "Forecasting",
  ),
  "education": (
    "Curriculum Development", "Lesson Planning", "Classroom Management",
    "Student Assessment", "Learning Management Systems",
  ),
  "sales": (
    "CRM", "Salesforce", "Lead Generation", "Negotiation", "Pipeline Management",
    "Client Relationship Management",
  ),
  "freshers": (
    "Git", "Python", "Java", "Teamwork", "Problem Solving", "Communication",
    "Microsoft Office", "Research",
  ),
}

_DEFAULT_ACTION_VERBS = (
  "Led", "Built", "Delivered", "Optimized", "Designed", "Implemented",
  "Automated", "Increased", "Reduced", "Collaborated", "Managed", "Developed",
  "Architected", "Streamlined", "Launched", "Integrated", "Mentored", "Resolved",
)

_VERB_RE = re.compile(
  r"\b(?:Built|Led|Designed|Optimized|Reduced|Increased|Developed|Implemented|"
  r"Managed|Delivered|Created|Automated|Collaborated|Launched|Integrated|"
  r"Architected|Streamlined|Mentored|Resolved|Negotiated|Closed|Analyzed)\b",
  re.I,
)

_SKILL_RE = re.compile(
  r"\b(?:Flutter|Dart|Python|Java|Kotlin|Firebase|React|Angular|Docker|AWS|"
  r"Git|SQL|PostgreSQL|MongoDB|GraphQL|Agile|Scrum|Figma|Excel|Salesforce)\b",
  re.I,
)

_LANG_KB_QUERIES: dict[str, str] = {
  "en": "resume professional summary en multilingual",
  "hi": "resume Hindi multilingual India",
  "es": "resume Spanish French German European",
  "fr": "resume Spanish French German European",
  "de": "resume Spanish French German European",
  "pt": "resume languages proficiency levels",
  "ar": "resume Arabic Middle East professional",
  "ja": "resume Japanese Korean Asian professional",
  "ko": "resume Japanese Korean Asian professional",
  "zh": "resume Japanese Korean Asian professional",
  "it": "resume Spanish French German European",
  "ru": "resume languages proficiency levels",
}


@lru_cache(maxsize=1)
def get_resume_kb() -> KnowledgeBase:
  return load_knowledge_base(knowledge_path=RESUME_KB_PATH)


def kb_search(query: str, limit: int = 3) -> list[dict[str, Any]]:
  kb = get_resume_kb()
  hits: list[dict[str, Any]] = []
  for answer, score in kb.search_ranked(query, limit=limit):
    if score < 0.08:
      continue
    hits.append({"text": answer, "score": round(score, 3), "source": "resume_knowledge"})
  return hits


def _pick(pool: list[str] | tuple[str, ...], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _extract_verbs_from_text(text: str) -> list[str]:
  return list(dict.fromkeys(m.group(0).title() for m in _VERB_RE.finditer(text or "")))


def _extract_skills_from_text(text: str) -> list[str]:
  return list(dict.fromkeys(m.group(0) if m.group(0).isupper() else m.group(0).title() for m in _SKILL_RE.finditer(text or "")))


def get_language_word_bank(lang_code: str | None) -> dict[str, Any]:
  """Open KB snippets for multilingual resume writing."""
  code = reng.bcp47(lang_code) if lang_code else "en"
  query = _LANG_KB_QUERIES.get(code, _LANG_KB_QUERIES["en"])
  hits = kb_search(query, limit=2)
  if code != "en":
    hits.extend(kb_search(f"resume {code} multilingual", limit=1))
  labels = reng.section_labels(lang_code)
  return {
    "code": code,
    "section_labels": labels,
    "kb_snippets": [h["text"] for h in hits],
    "summary_hint": hits[0]["text"][:280] if hits else "",
    "sources": ["resume_knowledge", "multilingual_word_bank"],
  }


def occupation_skill_bank(job_title: str, category: str, seed: int) -> list[str]:
  cat = category if category in _ESCO_TECH_SKILLS else "technology"
  base = list(_ESCO_TECH_SKILLS.get(cat, _ESCO_TECH_SKILLS["technology"]))
  hits = kb_search(f"{job_title} resume skills {cat}", limit=2)
  for hit in hits:
    base.extend(_extract_skills_from_text(hit["text"]))
    for part in re.split(r"[,;\n]", hit["text"]):
      token = part.strip()
      if 2 < len(token) < 40 and token[0].isupper():
        base.append(token)
  onet = list(_ONET_SOFT_SKILLS)
  for i in range(3):
    s = _pick(onet, seed + i)
    if s not in base:
      base.append(s)
  return list(dict.fromkeys(base))[:24]


def action_verb_bank(job_title: str, seed: int, docs: list[OpenDoc] | None = None) -> list[str]:
  verbs = list(_DEFAULT_ACTION_VERBS)
  hits = kb_search(f"{job_title} resume work experience bullet points", limit=2)
  for hit in hits:
    verbs.extend(_extract_verbs_from_text(hit["text"]))
  for doc in docs or []:
    verbs.extend(_extract_verbs_from_text(doc.text))
  return list(dict.fromkeys(verbs))[:20]


def pick_action_verb(seed: int, verbs: list[str] | None = None) -> str:
  pool = verbs or list(_DEFAULT_ACTION_VERBS)
  return _pick(pool, seed)


def enrich_skills_from_open_data(
  job_title: str,
  existing: list[str],
  *,
  category: str,
  docs: list[OpenDoc],
  seed: int,
) -> list[str]:
  merged = list(existing)
  merged.extend(occupation_skill_bank(job_title, category, seed))
  for doc in docs:
    merged.extend(_extract_skills_from_text(doc.text))
  for kw in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{1,24}", job_title):
    if len(kw) > 2:
      merged.append(kw)
  return list(dict.fromkeys(s.strip() for s in merged if s and len(s) > 1))[:22]


def summary_phrases_from_open_data(
  job_title: str,
  language: str | None,
  docs: list[OpenDoc],
) -> list[str]:
  phrases: list[str] = []
  lang_bank = get_language_word_bank(language)
  phrases.extend(lang_bank.get("kb_snippets") or [])
  phrases.extend(kb_search(f"{job_title} resume professional summary", limit=2))
  phrases = [p["text"] if isinstance(p, dict) else p for p in phrases]
  for doc in docs[:2]:
    snippet = re.sub(r"\s+", " ", doc.text)[:200]
    if snippet:
      phrases.append(snippet)
  return phrases[:5]


async def retrieve_resume_context(
  *,
  job_title: str,
  skills: list[str],
  language: str | None,
  category: str,
  seed: int,
  use_rag: bool = True,
) -> dict[str, Any]:
  """Fetch open datasets + local KB word banks for one resume request."""
  lang_code = reng.bcp47(language)
  lang_bank = get_language_word_bank(language)
  kb_role = kb_search(f"{job_title} resume best practices {category}", limit=3)
  docs: list[OpenDoc] = []

  if use_rag:
    lang_topic = f"resume CV {job_title} professional"
    if lang_code != "en":
      lang_topic = f"resume {reng.section_labels(language).get('summary', 'Summary')} {job_title} {lang_code}"
    try:
      docs = await asyncio.wait_for(
        retrieve_from_sources(
          lang_topic,
          [job_title, category, *skills[:3]],
          _RESUME_SOURCE_ROUTE,
          per_source=1,
          seed=seed,
        ),
        timeout=5.0,
      )
    except asyncio.TimeoutError:
      docs = []

  verbs = action_verb_bank(job_title, seed, docs)
  enriched_skills = enrich_skills_from_open_data(
    job_title, skills, category=category, docs=docs, seed=seed,
  )
  summary_phrases = summary_phrases_from_open_data(job_title, language, docs)

  sources = sorted({d.source for d in docs} | {"resume_knowledge", "esco", "onet"})
  return {
    "sources": sources,
    "documents": docs,
    "kb_hits": kb_role,
    "language_bank": lang_bank,
    "action_verbs": verbs,
    "skills": enriched_skills,
    "summary_phrases": summary_phrases,
    "onet_soft_skills": list(_ONET_SOFT_SKILLS[:8]),
    "esco_category": category,
  }
