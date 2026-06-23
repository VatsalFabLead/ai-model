"""Template-based resume/CV generator (pure code, no AI model).

Turns user details (free text or structured) into a clean, formatted resume in
markdown. 100% custom and free, with no third-party models involved.
"""

from __future__ import annotations

import re

_RESUME_TRIGGERS = (
  "resume", "cv", "curriculum vitae", "biodata", "bio data",
)
_INTENT_VERBS = ("make", "create", "build", "generate", "write", "need", "want", "prepare")


def detect_resume_intent(text: str) -> bool:
  low = text.lower()
  has_subject = any(t in low for t in _RESUME_TRIGGERS)
  has_verb = any(v in low for v in _INTENT_VERBS)
  return has_subject and has_verb


def _extract_years(text: str) -> str | None:
  m = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|year|yrs|yr)", text, re.IGNORECASE)
  if m:
    return m.group(1)
  return None


_ROLE_FILLER = {"am", "a", "an", "the", "i", "m", "is", "as", "my", "name", "are", "you"}


def _extract_role(text: str) -> str | None:
  # e.g. "flutter developer", "python engineer", "react developer"
  m = re.search(
    r"\b([A-Za-z.+#]+(?:\s+[A-Za-z.+#]+)?)\s+(developer|engineer|designer|"
    r"programmer|analyst|manager|consultant|architect|specialist)\b",
    text,
    re.IGNORECASE,
  )
  if not m:
    return None
  prefix_words = [w for w in m.group(1).split() if w.lower() not in _ROLE_FILLER]
  prefix = " ".join(prefix_words[-1:])  # keep the technology word closest to the role
  role_word = m.group(2)
  return (f"{prefix} {role_word}".strip()).title()


def _extract_name(text: str) -> str | None:
  m = re.search(r"\b(?:my name is|i am|i'm|name[:=])\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", text)
  if m:
    candidate = m.group(1).strip()
    # Avoid capturing role words like "a Flutter"
    if candidate.lower().split()[0] not in {"a", "an", "the"}:
      return candidate.title()
  return None


_KNOWN_SKILLS = {
  "flutter": ["Flutter", "Dart", "Firebase", "REST APIs", "Provider/Bloc", "Git"],
  "python": ["Python", "FastAPI", "Django", "Pandas", "REST APIs", "Git"],
  "react": ["React", "JavaScript", "TypeScript", "Redux", "HTML/CSS", "Git"],
  "android": ["Kotlin", "Java", "Android SDK", "Jetpack Compose", "REST APIs", "Git"],
  "node": ["Node.js", "Express", "JavaScript", "MongoDB", "REST APIs", "Git"],
  "java": ["Java", "Spring Boot", "Hibernate", "SQL", "REST APIs", "Git"],
}


def _skills_for(role: str | None, text: str) -> list[str]:
  low = (role or "").lower() + " " + text.lower()
  for key, skills in _KNOWN_SKILLS.items():
    if key in low:
      return skills
  return ["Problem Solving", "Communication", "Teamwork", "Git", "Time Management"]


def generate_resume(text: str) -> str:
  name = _extract_name(text) or "Your Name"
  role = _extract_role(text) or "Software Developer"
  years = _extract_years(text)
  skills = _skills_for(role, text)

  exp_line = (
    f"{years} years of professional experience as a {role}."
    if years
    else f"Experienced {role}."
  )
  years_label = f"{years} yrs" if years else "—"

  skills_md = "\n".join(f"- {s}" for s in skills)

  resume = f"""# {name}
**{role}**  |  📧 your.email@example.com  |  📱 +00 00000 00000  |  🌐 linkedin.com/in/you  |  💻 github.com/you

---

## Professional Summary
{exp_line} Passionate about building high-quality, user-friendly applications and continuously learning new technologies. Strong focus on clean code, performance, and collaboration.

---

## Skills
{skills_md}

---

## Experience

### {role} — Company Name
*Location · {years_label}*
- Built and shipped features end-to-end, improving app quality and user experience.
- Collaborated with designers and backend teams to deliver projects on time.
- Wrote clean, maintainable, and well-tested code following best practices.
- Optimized performance and fixed critical bugs to improve stability.

---

## Education

### Degree (e.g., B.E. / B.Tech / B.Sc) — University Name
*Year – Year*

---

## Projects
- **Project One** — short description, your role, and the tech you used.
- **Project Two** — short description, your role, and the tech you used.

---

_Tip: Replace the placeholder contact details, company, education, and projects with your real information. Want this exported as a PDF or as a Flutter CV app? Just ask._
"""
  return resume.strip()
