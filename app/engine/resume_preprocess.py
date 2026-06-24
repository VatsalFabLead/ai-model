"""Resume input preprocessing — spell correction, skill normalization, language validation."""

from __future__ import annotations

import difflib
import re
from typing import Any

from app.engine import resume_engine as reng

_TEXT_FIELDS = (
  "full_name", "job_title", "email", "phone", "linkedin", "portfolio",
  "summary", "education", "experience", "skills", "projects",
  "certifications", "achievements", "languages",
)

_SPELL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
  (re.compile(r"\bteh\b", re.I), "the"),
  (re.compile(r"\brecieve\b", re.I), "receive"),
  (re.compile(r"\bexperiance\b", re.I), "experience"),
  (re.compile(r"\bdevelper\b", re.I), "developer"),
  (re.compile(r"\bprograming\b", re.I), "programming"),
  (re.compile(r"\benviroment\b", re.I), "environment"),
  (re.compile(r"\bmanagment\b", re.I), "management"),
  (re.compile(r"\bimplementaion\b", re.I), "implementation"),
  (re.compile(r"\bresponsibilty\b", re.I), "responsibility"),
  (re.compile(r"\bacheivement\b", re.I), "achievement"),
  (re.compile(r"\bacheivements\b", re.I), "achievements"),
  (re.compile(r"\bcollegue\b", re.I), "colleague"),
  (re.compile(r"\bdefinately\b", re.I), "definitely"),
  (re.compile(r"\boccured\b", re.I), "occurred"),
  (re.compile(r"\bseperate\b", re.I), "separate"),
  (re.compile(r"\buntill\b", re.I), "until"),
  (re.compile(r"\bi\s+am\b"), "I am"),
  (re.compile(r"\bi\s+have\b"), "I have"),
  (re.compile(r"\bi\s+worked\b"), "I worked"),
]

_SKILL_SYNONYMS: dict[str, str] = {
  "js": "JavaScript",
  "javascript": "JavaScript",
  "ts": "TypeScript",
  "typescript": "TypeScript",
  "reactjs": "React",
  "react.js": "React",
  "react js": "React",
  "node": "Node.js",
  "nodejs": "Node.js",
  "node.js": "Node.js",
  "vuejs": "Vue.js",
  "vue.js": "Vue.js",
  "nextjs": "Next.js",
  "next.js": "Next.js",
  "k8s": "Kubernetes",
  "kube": "Kubernetes",
  "postgres": "PostgreSQL",
  "postgresql": "PostgreSQL",
  "mongo": "MongoDB",
  "mongodb": "MongoDB",
  "aws": "AWS",
  "gcp": "Google Cloud",
  "azure": "Microsoft Azure",
  "flutter": "Flutter",
  "dart": "Dart",
  "firebase": "Firebase",
  "git": "Git",
  "github": "GitHub",
  "gitlab": "GitLab",
  "ci/cd": "CI/CD",
  "cicd": "CI/CD",
  "rest": "REST APIs",
  "rest api": "REST APIs",
  "rest apis": "REST APIs",
  "graphql": "GraphQL",
  "docker": "Docker",
  "kubernetes": "Kubernetes",
  "python": "Python",
  "java": "Java",
  "kotlin": "Kotlin",
  "swift": "Swift",
  "android": "Android",
  "ios": "iOS",
  "figma": "Figma",
  "agile": "Agile",
  "scrum": "Scrum",
  "jira": "Jira",
  "sql": "SQL",
  "mysql": "MySQL",
  "sqlite": "SQLite",
}

_SUPPORTED_LANG_NAMES = {lang["name"].lower() for lang in reng.supported_languages()}
_SUPPORTED_LANG_CODES = {lang["code"].lower() for lang in reng.supported_languages()}

_YEARS_EXPERIENCE_RE = re.compile(
  r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+experience)?",
  re.I,
)


def extract_years_experience(text: str) -> float | None:
  """Parse '1.5 years' as 1.5 — never the trailing digit alone."""
  if not (text or "").strip():
    return None
  m = _YEARS_EXPERIENCE_RE.search(text)
  if not m:
    return None
  try:
    return float(m.group(1))
  except ValueError:
    return None


def seniority_from_years(years: float | None) -> str:
  if years is None:
    return "mid"
  if years < 2:
    return "entry"
  if years < 5:
    return "mid"
  return "senior"


def format_years_phrase(years: float | None) -> str:
  if years is None:
    return ""
  label = int(years) if years == int(years) else years
  return f" with {label} years of hands-on experience"


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def spell_correct_text(text: str) -> tuple[str, list[str]]:
  if not (text or "").strip():
    return "", []
  fixes: list[str] = []
  out = text
  for pat, repl in _SPELL_PATTERNS:
    if pat.search(out):
      fixes.append(f"{pat.pattern}→{repl}")
      out = pat.sub(repl, out)
  out = re.sub(r"[ \t]+", " ", out)
  out = re.sub(r"\s+([,.;:!?])", r"\1", out)
  return out, fixes


def spell_correct_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  corrected: dict[str, Any] = dict(payload)
  all_fixes: dict[str, list[str]] = {}
  for key in _TEXT_FIELDS:
    raw = payload.get(key)
    if raw is None or not str(raw).strip():
      continue
    fixed, fixes = spell_correct_text(str(raw))
    corrected[key] = fixed
    if fixes:
      all_fixes[key] = fixes
  return corrected, {
    "fields_corrected": list(all_fixes.keys()),
    "fix_count": sum(len(v) for v in all_fixes.values()),
    "fixes": all_fixes,
  }


def normalize_skill_token(token: str) -> str:
  t = _clean(token)
  if not t:
    return ""
  key = t.lower().replace(".", "").strip()
  if key in _SKILL_SYNONYMS:
    return _SKILL_SYNONYMS[key]
  if key in _SKILL_SYNONYMS.values():
    return next(v for v in _SKILL_SYNONYMS.values() if v.lower() == key)
  if len(t) <= 4 and t.isupper():
    return t.upper()
  if " " in t:
    return " ".join(w.capitalize() if w.lower() not in ("api", "apis", "ci", "cd") else w.upper() for w in t.split())
  return t[:1].upper() + t[1:] if t.islower() else t


def normalize_skills_text(text: str) -> tuple[str, list[str]]:
  if not (text or "").strip():
    return "", []
  parts = [p.strip() for p in re.split(r"[,;\n]+", text) if p.strip()]
  normalized = [normalize_skill_token(p) for p in parts]
  normalized = [n for n in normalized if n]
  return ", ".join(dict.fromkeys(normalized)), normalized


def normalize_skills_list(skills: list[str]) -> list[str]:
  out: list[str] = []
  seen: set[str] = set()
  for s in skills:
    n = normalize_skill_token(s)
    key = n.lower()
    if n and key not in seen:
      seen.add(key)
      out.append(n)
  return out


from app.engine.resume_open_data import get_language_word_bank

_EXTRA_SPOKEN_LANGUAGES = (
  "Gujarati", "Punjabi", "Telugu", "Kannada", "Malayalam", "Odia", "Assamese",
  "Nepali", "Sindhi", "Kashmiri", "Bhojpuri", "Maithili", "Sanskrit", "Latin",
  "Greek", "Hebrew", "Persian", "Farsi", "Swahili", "Filipino", "Tagalog",
  "Malay", "Romanian", "Czech", "Hungarian", "Swedish", "Norwegian", "Danish",
  "Finnish", "Ukrainian", "Catalan", "Serbian", "Croatian", "Slovak", "Bulgarian",
)

_SPOKEN_LANG_ALIASES: dict[str, str] = {
  "ennglish": "English", "engish": "English", "englsh": "English", "inglish": "English",
  "englis": "English", "englisch": "English", "eng": "English", "en": "English",
  "hindhi": "Hindi", "hindy": "Hindi", "hinfi": "Hindi", "hind": "Hindi",
  "gujrati": "Gujarati", "gujrathi": "Gujarati", "gujarathi": "Gujarati",
  "espanol": "Spanish", "español": "Spanish", "spanisch": "Spanish",
  "deutsch": "German", "allemand": "French", "francais": "French", "français": "French",
  "mandarin": "Chinese", "cantonese": "Chinese", "portuguese": "Portuguese",
  "bengali": "Bengali", "bangla": "Bengali", "marathi": "Marathi", "tamil": "Tamil",
  "telugu": "Telugu", "kannada": "Kannada", "malayalam": "Malayalam", "urdu": "Urdu",
  "punjabi": "Punjabi", "odia": "Odia", "oriya": "Odia",
}

_SPOKEN_LANG_BLOCKLIST = frozenset({
  "horny", "sexy", "love", "hate", "yes", "no", "ok", "test", "none", "na", "n/a",
  "good", "bad", "male", "female", "other", "unknown", "null", "nil",
})

_PROFICIENCY_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
  (re.compile(r"\bnative\b", re.I), "Native"),
  (re.compile(r"\bmother\s*tongue\b", re.I), "Native"),
  (re.compile(r"\bfluent\b", re.I), "Fluent"),
  (re.compile(r"\bprofessional\b", re.I), "Professional Working Proficiency"),
  (re.compile(r"\bworking\b", re.I), "Professional Working Proficiency"),
  (re.compile(r"\bconversational\b", re.I), "Conversational"),
  (re.compile(r"\bintermediate\b", re.I), "Intermediate"),
  (re.compile(r"\bbasic\b", re.I), "Basic"),
  (re.compile(r"\bbeginner\b", re.I), "Beginner"),
  (re.compile(r"\bcefr\s*[abc][12]\b", re.I), "CEFR Rated"),
)

_INDIAN_NATIVE_DEFAULT = frozenset({
  "Hindi", "Gujarati", "Bengali", "Tamil", "Marathi", "Urdu", "Telugu", "Kannada",
  "Malayalam", "Punjabi", "Odia", "Assamese", "Nepali", "Sindhi", "Kashmiri",
  "Bhojpuri", "Maithili", "Sanskrit",
})


def _spoken_language_catalog() -> tuple[dict[str, str], list[str]]:
  lookup: dict[str, str] = {}
  names: list[str] = []
  for lang in reng.supported_languages():
    name = lang["name"]
    names.append(name)
    lookup[name.lower()] = name
  for name in _EXTRA_SPOKEN_LANGUAGES:
    if name not in names:
      names.append(name)
    lookup[name.lower()] = name
  for alias, target in _SPOKEN_LANG_ALIASES.items():
    lookup[alias.lower()] = target
  return lookup, sorted(names, key=str.lower)


_SPOKEN_LANG_LOOKUP, _SPOKEN_LANG_NAMES = _spoken_language_catalog()


def _extract_proficiency(text: str) -> tuple[str, str | None]:
  """Return (language fragment, proficiency label)."""
  prof: str | None = None
  fragment = text
  for pat, label in _PROFICIENCY_MAP:
    if pat.search(fragment):
      prof = label
      fragment = pat.sub("", fragment)
  fragment = re.sub(r"[-–—|/]+", " ", fragment)
  fragment = re.sub(r"[()]", " ", fragment)
  return _clean(fragment), prof


def _resolve_spoken_language(token: str) -> str | None:
  key = _clean(token).lower()
  if not key or key in _SPOKEN_LANG_BLOCKLIST:
    return None
  if key in _SPOKEN_LANG_LOOKUP:
    return _SPOKEN_LANG_LOOKUP[key]
  if len(key) < 3:
    return None
  close = difflib.get_close_matches(key, [n.lower() for n in _SPOKEN_LANG_NAMES], n=1, cutoff=0.84)
  if close:
    return _SPOKEN_LANG_LOOKUP.get(close[0])
  return None


def _default_proficiency(language: str) -> str:
  if language == "English":
    return "Professional Working Proficiency"
  if language in _INDIAN_NATIVE_DEFAULT:
    return "Native Proficiency"
  return "Conversational"


def parse_spoken_languages(text: str) -> tuple[list[dict[str, str]], list[str]]:
  """Validate spoken languages against catalog; return entries and rejected tokens."""
  if not (text or "").strip():
    return [], []
  items = [p.strip() for p in re.split(r"[,;\n]+", text) if p.strip()]
  validated: list[dict[str, str]] = []
  rejected: list[str] = []
  seen: set[str] = set()
  for raw in items:
    fragment, prof = _extract_proficiency(raw)
    canon = _resolve_spoken_language(fragment)
    if not canon:
      rejected.append(raw)
      continue
    key = canon.lower()
    if key in seen:
      continue
    seen.add(key)
    validated.append({
      "language": canon,
      "proficiency": prof or _default_proficiency(canon),
    })
  return validated, rejected


def normalize_languages_text(text: str) -> tuple[str, dict[str, Any]]:
  """Comma-separated canonical language names for downstream fields."""
  entries, rejected = parse_spoken_languages(text)
  normalized = ", ".join(e["language"] for e in entries)
  return normalized, {
    "validated": entries,
    "rejected": rejected,
    "count": len(entries),
  }


def format_languages_section(text: str) -> tuple[str, dict[str, Any]]:
  """Resume Languages section with proficiency labels."""
  entries, rejected = parse_spoken_languages(text)
  lines = [f"- {e['language']} – {e['proficiency']}" for e in entries]
  return "\n".join(lines), {
    "validated": entries,
    "rejected": rejected,
    "count": len(entries),
  }


def validate_language(language: str | None, payload: dict[str, Any]) -> dict[str, Any]:
  if not language:
    bank = get_language_word_bank(None)
    return {
      "valid": True,
      "language": None,
      "code": reng.bcp47(None),
      "warnings": [],
      "word_bank": bank,
      "section_labels": bank.get("section_labels") or {},
    }
  raw = language.strip()
  low = raw.lower()
  warnings: list[str] = []
  bank = get_language_word_bank(raw)
  if low in _SUPPORTED_LANG_NAMES or low in _SUPPORTED_LANG_CODES:
    code = reng.bcp47(raw)
    return {
      "valid": True,
      "language": raw,
      "code": code,
      "warnings": warnings,
      "word_bank": bank,
      "section_labels": bank.get("section_labels") or {},
      "kb_snippets": bank.get("kb_snippets") or [],
    }
  warnings.append(f"unsupported_language:{raw}")
  return {
    "valid": True,
    "language": raw,
    "code": reng.bcp47(raw),
    "warnings": warnings,
    "fallback": "en",
    "word_bank": bank,
    "section_labels": bank.get("section_labels") or {},
  }
