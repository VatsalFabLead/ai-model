"""Deterministic resume/CV generator.

100% custom and free — pure Python, no AI model. Detects a "build my resume"
request, extracts details (role, years), and fills a professional template.
"""

from __future__ import annotations

import re

_ROLE_RE = re.compile(
  r"\b((?:[a-z\+\#\.]+\s+){0,2}"
  r"(?:developer|engineer|designer|programmer|analyst|scientist|manager|"
  r"administrator|architect|consultant|specialist|tester|marketer|writer))\b",
  re.IGNORECASE,
)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)", re.IGNORECASE)

_SKILLS_BY_ROLE = {
  "flutter": [
    "Flutter", "Dart", "REST APIs", "Firebase", "State management (Provider/Bloc/Riverpod)",
    "SQLite/Hive", "Git & GitHub", "CI/CD", "Material & Cupertino UI", "Push notifications",
  ],
  "python": [
    "Python", "FastAPI/Django/Flask", "REST APIs", "PostgreSQL/MySQL", "Pandas/NumPy",
    "Docker", "Git & GitHub", "Unit testing", "Linux", "Cloud (AWS/GCP)",
  ],
  "react": [
    "React", "JavaScript/TypeScript", "Redux", "HTML5 & CSS3", "REST/GraphQL APIs",
    "Tailwind CSS", "Jest", "Git & GitHub", "Webpack/Vite", "Responsive design",
  ],
  "android": [
    "Kotlin", "Java", "Android SDK", "Jetpack Compose", "MVVM", "Retrofit",
    "Room", "Coroutines", "Git & GitHub", "Material Design",
  ],
  "web": [
    "HTML5", "CSS3", "JavaScript", "React/Vue", "REST APIs", "Responsive design",
    "Git & GitHub", "Webpack/Vite", "Accessibility", "Cross-browser testing",
  ],
}
_DEFAULT_SKILLS = [
  "[Skill 1]", "[Skill 2]", "[Skill 3]", "[Skill 4]", "[Skill 5]",
  "Git & GitHub", "Problem solving", "Teamwork", "Communication",
]


def _wants_resume(text: str) -> bool:
  low = text.lower()
  has_resume = any(w in low for w in ("resume", "cv", "curriculum vitae"))
  is_personal = any(
    w in low
    for w in ("i am", "i'm", "im a", "my ", "myself", "experience", "year", "yr", "fresher")
  )
  wants_build = any(
    w in low for w in ("make", "build", "create", "generate", "write", "want", "need", "prepare")
  )
  return has_resume and is_personal and wants_build


_ROLE_FILLER = {"am", "a", "an", "the", "i", "im", "is", "are", "m", "as", "of", "experienced"}


def _detect_role(text: str) -> str | None:
  m = _ROLE_RE.search(text)
  if not m:
    return None
  words = re.sub(r"\s+", " ", m.group(1).strip()).split()
  while words and words[0].lower() in _ROLE_FILLER:
    words.pop(0)
  if not words:
    return None
  return " ".join(words).title()


def _skills_for(role: str | None) -> list[str]:
  if not role:
    return _DEFAULT_SKILLS
  low = role.lower()
  for key, skills in _SKILLS_BY_ROLE.items():
    if key in low:
      return skills
  return _DEFAULT_SKILLS


def generate_resume(text: str) -> str | None:
  """Return a filled resume if the message asks to build one, else None."""
  if not _wants_resume(text):
    return None

  role = _detect_role(text)
  years_m = _YEARS_RE.search(text)
  years = years_m.group(1) if years_m else None

  role_title = role or "[Your Job Title]"
  exp_line = f"{years} Years of Experience" if years else "[X] Years of Experience"
  skills = _skills_for(role)
  skills_block = "\n".join(f"- {s}" for s in skills)

  if years:
    summary = (
      f"{role_title} with {years} years of hands-on experience building and shipping "
      f"high-quality applications. Skilled at writing clean, maintainable code, "
      f"collaborating with teams, and delivering features that improve user experience "
      f"and performance. Seeking to contribute to impactful products."
    )
  else:
    summary = (
      f"{role_title} passionate about building high-quality applications and writing "
      f"clean, maintainable code. [Add 1-2 lines about your strengths and goals.]"
    )

  return f"""Here is a ready-to-use resume draft based on your details. Replace the [bracketed]
placeholders with your information.

---

# [Your Full Name]
**{role_title}** | {exp_line}
[email@example.com] • [+91-XXXXXXXXXX] • [City, Country]
[LinkedIn URL] • [GitHub/Portfolio URL]

## Professional Summary
{summary}

## Skills
{skills_block}

## Experience
**{role_title}** — [Company Name]  ([Start Month Year] – [Present])
- [Built/Delivered ... that resulted in ... — add a number or impact]
- [Implemented ... using ... improving ... by X%]
- [Collaborated with ... to ...]

## Education
**[Degree, e.g., B.Tech in Computer Science]** — [Institution Name]  ([Year])

## Projects
**[Project Name]** — [Tech stack]
- [What it does, your role, and a link]

---

Tips for a strong resume:
- Keep it to 1-2 pages and use consistent formatting.
- Start each bullet with an action verb (Built, Designed, Improved, Led).
- Add numbers/impact wherever possible (users, performance, revenue, time saved).
- Tailor the summary and skills to each job you apply for.

Want me to turn this into a Flutter app or a downloadable PDF layout?"""
