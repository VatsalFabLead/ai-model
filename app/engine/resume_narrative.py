"""Convert conversational first-person resume input into professional third-person output."""

from __future__ import annotations

import re
from typing import Any

from app.engine.resume_preprocess import (
  extract_years_experience,
  format_languages_section,
  format_years_phrase,
)

_ACTION_VERBS = (
  "Developed", "Built", "Designed", "Implemented", "Delivered", "Optimized",
  "Created", "Integrated", "Collaborated on", "Maintained",
)

_SKILL_CATALOG = (
  "Flutter", "Dart", "Python", "Java", "Kotlin", "Firebase", "React", "Angular",
  "Android", "iOS", "REST APIs", "GraphQL", "Git", "GitHub",
  "MySQL", "SQLite", "PostgreSQL", "GetX", "Provider", "MVVM",
  "Android Studio", "VS Code", "Postman", "JavaScript", "TypeScript", "Node.js",
)

_RESUME_ACTION_VERBS = frozenset(
  v.lower() for v in (
    *_ACTION_VERBS,
    "Led", "Managed", "Resolved", "Automated", "Improved", "Reduced", "Increased",
    "Collaborated", "Fixed", "Used", "Worked", "Supported", "Enhanced",
  )
)

_META_EXPERIENCE_RE = re.compile(
  r"^(?:\d+(?:\.\d+)?\s*\+?\s*years?(?:\s+of)?(?:\s+experience)?|"
  r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,]+\d{4}|"
  r"\d{4}\s*[–\-]\s*(?:present|current|\w+)|present|current)$",
  re.I,
)

_FIRST_PERSON_RE = re.compile(
  r"^(?:I am|I'm|I have|I've|I was|I work|I worked|I use|I also|I can|I want|"
  r"I completed|I created|I built|I developed|I made|I finished|I studied|"
  r"I learned|I always|I integrate|I fixed|I added|I maintained)\s+",
  re.I,
)


def _pick(pool: tuple[str, ...], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


_FIRST_PERSON_REPLACEMENTS = (
  (re.compile(r"^I work on\s+", re.I), "Worked on "),
  (re.compile(r"^I worked on\s+", re.I), "Worked on "),
  (re.compile(r"^I integrate\s+", re.I), "Integrated "),
  (re.compile(r"^I integrated\s+", re.I), "Integrated "),
  (re.compile(r"^I develop(?:ed)?\s+", re.I), "Developed "),
  (re.compile(r"^I built\s+", re.I), "Built "),
  (re.compile(r"^I created\s+", re.I), "Created "),
  (re.compile(r"^I fixed\s+", re.I), "Resolved "),
  (re.compile(r"^I added\s+", re.I), "Added "),
  (re.compile(r"^I maintained\s+", re.I), "Maintained "),
  (re.compile(r"^I also\s+", re.I), ""),
)


def de_first_person(sentence: str) -> str:
  s = (sentence or "").strip()
  for pat, repl in _FIRST_PERSON_REPLACEMENTS:
    s = pat.sub(repl, s)
  for _ in range(3):
    s2 = _FIRST_PERSON_RE.sub("", s).strip()
    s2 = re.sub(r"^I\s+", "", s2, flags=re.I).strip()
    if s2 == s:
      break
    s = s2
  s = re.sub(r"\bmy\b", "the", s, flags=re.I)
  s = re.sub(r"\bme\b", "the team", s, flags=re.I)
  return s.strip()


def extract_skills_from_narrative(text: str) -> list[str]:
  if not (text or "").strip():
    return []
  low = text.lower()
  found: list[str] = []
  for skill in _SKILL_CATALOG:
    if re.search(rf"\b{re.escape(skill)}\b", low, re.I) and skill not in found:
      found.append(skill)
  for m in re.finditer(
    r"(?:experience with|skilled in|proficient in|using|use)\s+([^.]+)",
    text,
    re.I,
  ):
    for part in re.split(r",|\band\b|/|;", m.group(1)):
      token = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", part.strip())
      token = re.sub(
        r"^(?:I have|I use|I am familiar with|I worked with)\s+",
        "",
        token,
        flags=re.I,
      ).strip()
      if 2 < len(token) < 36 and token.lower() not in {"and", "the", "for", "with"}:
        found.append(token.title() if " " not in token else token)
  return list(dict.fromkeys(found))


def _sentences(text: str) -> list[str]:
  return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if len(s.strip()) > 12]


def is_meta_experience_line(line: str) -> bool:
  ln = (line or "").strip()
  if not ln:
    return True
  if _META_EXPERIENCE_RE.match(ln):
    return True
  if re.fullmatch(r"\d+(?:\.\d+)?\s*\+?\s*years?", ln, re.I):
    return True
  return False


def line_has_action_verb(line: str) -> bool:
  first = (line.split() or [""])[0].rstrip(".,;:-")
  return first.lower() in _RESUME_ACTION_VERBS


def is_minimal_experience(text: str) -> bool:
  lines = [ln for ln in re.split(r"[\n;]+", text or "") if ln.strip()]
  if not lines:
    return True
  content_lines = [ln.strip() for ln in lines if not is_meta_experience_line(ln.strip())]
  return len(content_lines) == 0


def _domain_work_phrase(domain: str, title: str, skills: list[str]) -> str:
  low = f"{title} {domain}".lower()
  if "flutter" in low or "mobile" in low:
    return "developing cross-platform mobile applications using Flutter and Dart"
  if "data" in low or "ml" in low:
    return "building data-driven solutions and analytics workflows"
  if "design" in low or "ux" in low:
    return "designing intuitive user experiences and visual systems"
  lead = skills[0] if skills else title
  return f"delivering high-quality work as a {title} with strong command of {lead}"


def build_role_experience_bullets(job_title: str, skills: list[str]) -> list[str]:
  low = (job_title or "").lower()
  stack = ", ".join(skills[:4]) if skills else "modern development tools"
  if "flutter" in low or "mobile" in low:
    return [
      "Developed and maintained cross-platform mobile applications using Flutter and Dart.",
      "Integrated REST APIs and Firebase services for authentication and real-time data synchronization.",
      "Collaborated with backend developers and designers to deliver user-friendly applications.",
      "Fixed bugs and optimized application performance to improve user experience.",
      "Used Git for version control and participated in code reviews.",
    ]
  return [
    f"Developed and delivered features as {job_title} using {stack}.",
    "Collaborated with cross-functional teams to plan, build, test, and release software.",
    "Resolved defects and improved application quality through debugging and code reviews.",
    "Applied version control and documentation best practices for maintainable delivery.",
  ]


def parse_experience_header(text: str, job_title: str) -> tuple[str, str, str]:
  raw = (text or "").strip()
  company_m = re.search(
    r"(?:at|@)\s+([A-Z][A-Za-z0-9&.,'()\- ]{2,80}(?:"
    r"Pvt\.?\s*Ltd\.?|Ltd\.?|LLC|Inc\.?|Technologies|Solutions|Corp\.?)?)",
    raw,
    re.I,
  )
  date_m = re.search(
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r"(?:\s*[–\-]\s*(?:Present|Current|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}))?)",
    raw,
    re.I,
  )
  role_m = re.search(
    r"(?:as|working as|worked as|position[:\s]+)\s*(?:a\s+)?([^,.|]+)",
    raw,
    re.I,
  )
  role = (role_m.group(1).strip() if role_m else job_title).strip()
  company = (company_m.group(1).strip() if company_m else "").strip()
  dates = (date_m.group(1).strip() if date_m else "").strip()
  return role, company, dates


def format_structured_experience(
  text: str,
  job_title: str,
  skills: list[str],
  seed: int,
) -> tuple[str, list[str]]:
  role, company, dates = parse_experience_header(text, job_title)
  header_parts = [role]
  if company:
    header_parts.append(company)
  if dates:
    header_parts.append(dates)
  header = "\n".join(header_parts)
  bullets = build_role_experience_bullets(role, skills)
  body = header + "\n\n" + "\n".join(f"- {b}" for b in bullets)
  return body, bullets


def rewrite_summary_narrative(
  existing: str,
  personal: dict[str, Any],
  skills: list[str],
  understanding: dict[str, Any] | None,
  seed: int,
) -> str:
  """Professional summary — recruiter tone, no third-person name, user skills only."""
  title = personal.get("job_title") or "Professional"
  skill_text = ", ".join(skills[:8]) if skills else "relevant technologies"
  domain = (understanding or {}).get("domain", "software engineering")
  years = (understanding or {}).get("years_experience")
  if years is None:
    years = extract_years_experience(existing or "")
  years_clause = ""
  if years is not None:
    label = int(years) if years == int(years) else years
    years_clause = f" with {label} years of experience"

  work_phrase = _domain_work_phrase(domain, title, skills)
  return (
    f"Results-driven {title}{years_clause} {work_phrase}. "
    f"Skilled in {skill_text}. "
    f"Experienced in building scalable applications, integrating cloud services, "
    f"and collaborating with cross-functional teams to deliver high-quality software solutions."
  )


def rewrite_experience_narrative(text: str, job_title: str, seed: int) -> tuple[str, list[str]]:
  raw = (text or "").strip()
  if not raw:
    return "", []
  if is_minimal_experience(raw):
    return format_structured_experience(raw, job_title, [], seed)

  blocks = re.split(
    r"\s*(?:Before that|Previously|Earlier|Prior to that)[,.]?\s*",
    raw,
    flags=re.I,
  )
  if len(blocks) <= 1:
    blocks = re.split(
      r"(?=\s*I\s+(?:have been|worked as|was a|am a)\s+)",
      raw,
      flags=re.I,
    )
  blocks = [b.strip() for b in blocks if b.strip()]

  sections: list[str] = []
  all_bullets: list[str] = []
  for bi, block in enumerate(blocks[:4]):
    title_m = re.search(
      r"(?:as|working as|been working as|worked as)\s+(?:a\s+)?([^,.]+?)"
      r"(?:\s+at|\s+since|\s+from|\.|,|$)",
      block,
      re.I,
    )
    company_m = re.search(
      r"at\s+([^,.]+?)(?:\s+since|\s+from|\s+in\s+\w|\s+during|\.|,|$)",
      block,
      re.I,
    )
    date_m = re.search(
      r"(?:since|from)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
      r"(?:\s*[–\-]\s*(?:Present|Current|\w+\s+\d{4}))?|\d{4}\s*[–\-]\s*\w+)",
      block,
      re.I,
    )

    role = (title_m.group(1).strip() if title_m else job_title).strip()
    company = (company_m.group(1).strip() if company_m else "").strip()
    dates = (date_m.group(1).strip() if date_m else "").strip()

    header_parts = [role]
    if company:
      header_parts.append(company)
    header = header_parts[0] if len(header_parts) == 1 else f"{header_parts[0]}\n{header_parts[1]}"
    if dates:
      header += f" | {dates}"

    bullets: list[str] = []
    for si, sent in enumerate(_sentences(block)):
      if re.search(r"(?:been working as|worked as|working as|I am a)\s", sent, re.I):
        continue
      sent = de_first_person(sent)
      if len(sent) < 22:
        continue
      if re.search(r"^(?:on|and|also|the)\b", sent, re.I):
        continue
      if is_meta_experience_line(sent):
        continue
      if not line_has_action_verb(sent):
        sent = f"Developed {sent[0].lower() + sent[1:]}" if sent else sent
      bullet = f"- {sent.rstrip('.')}."
      if bullet not in bullets:
        bullets.append(bullet)

    if len(bullets) < 2:
      for si, sent in enumerate(_sentences(block)):
        if re.search(r"(?:been working as|worked as|working as)\s", sent, re.I):
          continue
        sent = de_first_person(sent)
        if len(sent) < 18 or is_meta_experience_line(sent):
          continue
        line = f"- {sent.rstrip('.')}."
        if not line_has_action_verb(sent):
          line = f"- Developed {sent[0].lower() + sent[1:] if sent else sent}.".replace("..", ".")
        if line not in bullets:
          bullets.append(line)
        if len(bullets) >= 5:
          break

    bullets = bullets[:6]
    if not bullets:
      bullets = [f"- {b}" for b in build_role_experience_bullets(role, [])]

    section = header + "\n\n" + "\n".join(bullets)
    sections.append(section)
    all_bullets.extend(bullets)

  return "\n\n".join(sections), all_bullets


def rewrite_education_narrative(text: str) -> str:
  raw = (text or "").strip()
  if not raw:
    return ""
  degree_m = re.search(
    r"(Bachelor of Engineering[^.]*|Bachelor[^.]*(?:Engineering|Science|Technology)[^.]*|"
    r"B\.?\s*E\.?[^.]*|Master[^.]*|M\.?\s*Tech[^.]*)",
    raw,
    re.I,
  )
  school_m = re.search(
    r"(?:from|at)\s+([^,.]+(?:Institute|University|College|Technology)[^,.]*)",
    raw,
    re.I,
  )
  year_m = re.search(r"(?:graduat|completed|studied).*?(20\d{2})", raw, re.I)
  gpa_m = re.search(r"CGPA\s*(?:of\s*)?(\d+\.?\d*)", raw, re.I)
  coursework_m = re.search(
    r"(?:learned|studied|coursework)[^.]*(?:programming|database|software)[^.]*",
    raw,
    re.I,
  )

  lines: list[str] = []
  if degree_m:
    deg = degree_m.group(0).strip().rstrip(".")
    deg = re.sub(r"\s+from\s+.*$", "", deg, flags=re.I)
    lines.append(deg)
  elif re.search(r"computer engineering|computer science", raw, re.I):
    lines.append("Bachelor of Engineering in Computer Engineering")
  if school_m:
    school = school_m.group(1).strip()
    if school not in (lines[-1] if lines else ""):
      lines.append(school)
  meta: list[str] = []
  if year_m:
    meta.append(f"Graduation Year: {year_m.group(1)}")
  if gpa_m:
    meta.append(f"CGPA: {gpa_m.group(1)}")
  if meta:
    lines.append(" | ".join(meta))
  if coursework_m:
    lines.append(
      "Relevant Coursework: Programming, Databases, Software Development, Data Structures"
    )
  if lines:
    return "\n".join(lines)
  return "\n".join(f"- {de_first_person(s)}" for s in _sentences(raw)[:4])


def _project_paragraph(name: str, tech: str, job_title: str) -> str:
  low = name.lower()
  if "food" in low and "deliver" in low:
    return (
      f"Built a cross-platform food delivery application using {tech}. "
      "Implemented customer, restaurant, and delivery modules with real-time order tracking, "
      "notifications, and secure API integration."
    )
  if "e-commerce" in low or "ecommerce" in low or "commerce" in low:
    return (
      f"Developed an e-commerce application with product browsing, shopping cart, user authentication, "
      f"and payment integration using {tech}. Improved application performance and user experience "
      "through responsive UI design."
    )
  if "face" in low and "detect" in low:
    return (
      "Created a face detection application using machine learning techniques for image processing "
      "and facial recognition. Integrated camera functionality and optimized detection accuracy."
    )
  if "todo" in low:
    return (
      f"Built a task management application using {tech} with CRUD workflows, local persistence, "
      "and a responsive interface for daily productivity."
    )
  verb = "Built" if "app" in low else "Developed"
  return (
    f"{verb} {name} using {tech}. Delivered core features with a focus on performance, "
    "maintainability, and user experience."
  )


def rewrite_projects_narrative(
  text: str,
  job_title: str,
  skills: list[str],
  seed: int,
) -> str:
  raw = (text or "").strip()
  if not raw:
    return ""
  parts = re.split(
    r"(?=\s*I\s+(?:developed|built|created|made|designed)\s+)",
    raw,
    flags=re.I,
  )
  parts = [p.strip() for p in parts if p.strip()]
  if len(parts) <= 1:
    parts = _sentences(raw)

  skill_hint = ", ".join(skills[:5]) if skills else "relevant technologies"
  out: list[str] = []
  for i, part in enumerate(parts[:5]):
    part = de_first_person(part)
    title_m = re.match(
      r"(?:developed|built|created|made|designed)\s+(?:(?:an|a)\s+)?([^,.]+)",
      part,
      re.I,
    )
    title = (title_m.group(1).strip() if title_m else f"Project {i + 1}").strip()
    tech_m = re.findall(
      r"\b(Flutter|Dart|Firebase|Python|REST APIs?|MySQL|SQLite|Android|iOS|JavaScript|Node\.js)\b",
      part,
      re.I,
    )
    tech = ", ".join(
      dict.fromkeys(t.title() if t.lower() != "rest apis" else "REST APIs" for t in tech_m)
    ) or skill_hint
    title = re.sub(r"\s+using\s+.+$", "", title, flags=re.I).strip()
    paragraph = _project_paragraph(title, tech, job_title)
    out.append(f"**{title}**\n{paragraph}")
  return "\n\n".join(out)


def optimize_projects_structured(
  text: str,
  job_title: str,
  skills: list[str],
  seed: int,
) -> str:
  """Expand project lines into title + natural paragraph (no template verb spam)."""
  lines = []
  for ln in (text or "").splitlines():
    ln = re.sub(r"^[\-\*\•\d]+[\).\s]+", "", ln.strip())
    if ln:
      lines.append(ln)
  if not lines:
    return ""

  skill_hint = ", ".join(skills[:5]) if skills else "relevant technologies"
  out: list[str] = []
  for i, ln in enumerate(lines[:6]):
    if "|" in ln:
      name, tech = [p.strip() for p in ln.split("|", 1)]
    elif "—" in ln:
      name, tech = [p.strip() for p in ln.split("—", 1)]
    else:
      name, tech = ln.strip(), skill_hint
    name = re.sub(r"\s+using\s+.+$", "", name, flags=re.I).strip()
    paragraph = _project_paragraph(name, tech, job_title)
    out.append(f"**{name}**\n{paragraph}")
  return "\n\n".join(out)


def enhance_projects_section(
  text: str,
  job_title: str,
  skills: list[str],
  seed: int,
) -> str:
  raw = (text or "").strip()
  if not raw:
    return ""
  if is_narrative_text(raw):
    return rewrite_projects_narrative(raw, job_title, skills, seed)
  return optimize_projects_structured(raw, job_title, skills, seed)


def rewrite_certifications_narrative(text: str) -> str:
  raw = (text or "").strip()
  if not raw:
    return ""
  items: list[str] = []
  for m in re.finditer(
    r"(?:completed|finished)\s+(.+?)(?:\s+from\s+([A-Za-z]+))?(?:\.| and I| and also|$)",
    raw,
    re.I,
  ):
    name = m.group(1).strip().rstrip(".")
    provider = (m.group(2) or "Professional Development").strip()
    if len(name) > 5:
      items.append(f"- {name} – {provider}")
  if not items:
    for sent in _sentences(raw):
      sent = de_first_person(sent)
      if len(sent) > 10:
        items.append(f"- {sent.rstrip('.')}")
  return "\n".join(items[:8])


def rewrite_achievements_narrative(text: str, seed: int) -> str:
  raw = (text or "").strip()
  if not raw:
    return ""
  bullets: list[str] = []
  for i, sent in enumerate(_sentences(raw)):
    sent = de_first_person(sent)
    if len(sent) < 15:
      continue
    verb = _pick(_ACTION_VERBS, seed + i)
    if not re.match(r"^[A-Z]", sent):
      sent = f"{verb} {sent[0].lower() + sent[1:]}"
    bullets.append(f"- {sent.rstrip('.')}.")
  return "\n".join(bullets[:6])


def rewrite_languages_narrative(text: str) -> str:
  body, _ = format_languages_section(text)
  return body


def is_narrative_text(text: str) -> bool:
  low = (text or "").lower()
  return bool(
    re.search(r"\bi\s+(?:am|have|worked|completed|developed|built|can|use)\b", low)
    or len(_sentences(text)) <= 2 and len(text or "") > 120
  )
