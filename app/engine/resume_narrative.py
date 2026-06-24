"""Convert conversational first-person resume input into professional third-person output."""

from __future__ import annotations

import re
from typing import Any

_ACTION_VERBS = (
  "Led", "Built", "Delivered", "Optimized", "Designed", "Implemented",
  "Automated", "Developed", "Integrated", "Collaborated", "Managed", "Reduced",
)

_SKILL_CATALOG = (
  "Flutter", "Dart", "Python", "Java", "Kotlin", "Firebase", "React", "Angular",
  "Android", "iOS", "REST APIs", "GraphQL", "Git", "GitHub", "Docker", "AWS",
  "MySQL", "SQLite", "PostgreSQL", "GetX", "Provider", "MVVM", "Agile", "Scrum",
  "Android Studio", "VS Code", "Postman", "JavaScript", "TypeScript", "Node.js",
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
    if skill.lower() in low and skill not in found:
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


def rewrite_summary_narrative(
  existing: str,
  personal: dict[str, Any],
  skills: list[str],
  understanding: dict[str, Any] | None,
  seed: int,
) -> str:
  """Professional summary without echoing raw input."""
  title = personal.get("job_title") or "Professional"
  name = (personal.get("full_name") or "Candidate").split()[0]
  skill_text = ", ".join(skills[:6]) if skills else "modern software delivery"
  domain = (understanding or {}).get("domain", "software engineering")
  seniority = (understanding or {}).get("seniority", "mid")
  verb = _pick(_ACTION_VERBS, seed)
  years = ""
  if re.search(r"\b(2|3|4|5)\+?\s*years?\b", existing or "", re.I):
    years = " with 2+ years of hands-on experience"
  return (
    f"{verb} {title}{years} focused on {domain}. Expertise in {skill_text}. "
    f"Track record of shipping production features, integrating APIs and cloud services, "
    f"and collaborating with cross-functional teams. {name} brings {seniority}-level ownership, "
    f"clean code practices, and measurable impact for hiring workflows."
  )


def rewrite_experience_narrative(text: str, job_title: str, seed: int) -> tuple[str, list[str]]:
  raw = (text or "").strip()
  if not raw:
    return "", []

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
      verb = _pick(_ACTION_VERBS, seed + bi + si)
      if not re.match(r"^[A-Z]", sent):
        sent = f"{verb} {sent[0].lower() + sent[1:]}" if sent else sent
      bullet = f"- {sent.rstrip('.')}."
      if bullet not in bullets:
        bullets.append(bullet)

    if len(bullets) < 2:
      for si, sent in enumerate(_sentences(block)):
        if re.search(r"(?:been working as|worked as|working as)\s", sent, re.I):
          continue
        sent = de_first_person(sent)
        if len(sent) < 18:
          continue
        verb = _pick(_ACTION_VERBS, seed + bi + si + 5)
        line = f"- {verb} {sent[0].lower() + sent[1:] if sent else sent}.".replace("..", ".")
        if line not in bullets:
          bullets.append(line)
        if len(bullets) >= 5:
          break

    bullets = bullets[:6]
    if not bullets:
      bullets = [
        f"- {_pick(_ACTION_VERBS, seed)} cross-platform features using modern tooling.",
        f"- {_pick(_ACTION_VERBS, seed + 1)} APIs, cloud services, and version control in agile delivery.",
      ]

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


def rewrite_projects_narrative(text: str, seed: int) -> str:
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

  out: list[str] = []
  for i, part in enumerate(parts[:5]):
    part = de_first_person(part)
    title_m = re.match(
      r"(?:developed|built|created|made|designed)\s+(?:a|an)?\s*([^,.]+)",
      part,
      re.I,
    )
    title = (title_m.group(1).strip() if title_m else f"Project {i + 1}").strip()
    tech_m = re.findall(
      r"\b(Flutter|Dart|Firebase|Python|REST APIs?|MySQL|SQLite|Android|iOS)\b",
      part,
      re.I,
    )
    tech = ", ".join(dict.fromkeys(t.title() if t.lower() != "rest apis" else "REST APIs" for t in tech_m))
    verb = _pick(_ACTION_VERBS, seed + i)
    desc = part
    if title_m:
      desc = part[title_m.end():].strip(" .,-")
    if not desc:
      desc = part
    header = f"**{title}**" + (f" | {tech}" if tech else "")
    body = f"{verb} {desc[0].lower() + desc[1:] if desc and desc[0].isupper() else desc}.".replace("..", ".")
    out.append(f"{header}\n\n{body}")
  return "\n\n".join(out)


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
  raw = (text or "").strip()
  if not raw:
    return ""
  items: list[str] = []
  if re.search(r"english", raw, re.I):
    prof = "Native" if re.search(r"native english", raw, re.I) else "Professional Working Proficiency"
    items.append(f"- English – {prof}")
  if re.search(r"hindi", raw, re.I):
    items.append("- Hindi – Native Proficiency")
  if re.search(r"gujarati", raw, re.I):
    items.append("- Gujarati – Native Proficiency")
  if items:
    return "\n".join(items)
  return "\n".join(f"- {s.strip()}" for s in re.split(r"[,;\n]+", raw) if s.strip())


def is_narrative_text(text: str) -> bool:
  low = (text or "").lower()
  return bool(
    re.search(r"\bi\s+(?:am|have|worked|completed|developed|built|can|use)\b", low)
    or len(_sentences(text)) <= 2 and len(text or "") > 120
  )
