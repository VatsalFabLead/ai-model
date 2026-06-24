"""Combinatorial cover-letter templates — thousands of tone-aware variations per section.

Each section picks one phrase from several independent banks; combinations multiply
(e.g. 30 x 28 x 26 x 24 x 8 tones > 500,000 unique intros).
"""

from __future__ import annotations

import re
from typing import Any

VALID_TONES = (
  "professional", "casual", "friendly", "formal",
  "confident", "enthusiastic", "persuasive", "neutral",
)

# --- Shared banks (cross-tone) ------------------------------------------------

_GREETING_STEMS = [
  "Dear Hiring Manager,",
  "Dear Recruitment Team,",
  "Dear Talent Acquisition Team,",
  "Dear {company} Hiring Team,",
  "Dear {company} Recruitment Team,",
  "Dear {company} Talent Team,",
  "Dear {company} Team,",
  "To the Hiring Manager at {company},",
  "To the {company} Hiring Committee,",
  "Dear Sir or Madam,",
]

_INTRO_OPENERS = [
  "I am writing to apply for the {title} position at {company}.",
  "I am applying for the {title} role at {company}.",
  "I wish to express my interest in the {title} opening at {company}.",
  "I was pleased to discover the {title} opportunity at {company}.",
  "I am excited to submit my application for the {title} role at {company}.",
  "Please accept my application for the {title} position at {company}.",
  "I am reaching out regarding the {title} vacancy at {company}.",
  "I would like to be considered for the {title} role at {company}.",
  "I am motivated to pursue the {title} position with {company}.",
  "I am eager to apply for the {title} opportunity at {company}.",
  "Having followed {company}'s work, I am applying for the {title} role.",
  "The {title} posting at {company} caught my attention and I am applying today.",
  "I am confident I am a strong match for the {title} role at {company}.",
  "I am interested in joining {company} as a {title}.",
  "I am submitting my candidacy for the {title} position at {company}.",
  "I would welcome the chance to interview for the {title} role at {company}.",
  "My background aligns well with the {title} opening at {company}.",
  "I am drawn to the {title} opportunity at {company} and am applying now.",
  "I am pursuing the {title} position at {company} with enthusiasm.",
  "I believe I can add value in the {title} role at {company}.",
  "I am keen to contribute as a {title} at {company}.",
  "I am ready to bring my experience to the {title} team at {company}.",
  "I am applying because the {title} role at {company} fits my career direction.",
  "I see a strong fit between my experience and the {title} role at {company}.",
  "I am interested in the {title} position and the impact I could make at {company}.",
  "I would be honored to be considered for the {title} role at {company}.",
  "I am looking for my next challenge as a {title} and {company} stands out.",
  "I am applying for the {title} role where I can grow with {company}.",
  "I want to bring my skills to {company} in the {title} capacity.",
  "I am enthusiastic about the {title} opening listed by {company}.",
]

_INTRO_BRIDGES = [
  "Across my career, I have built practical depth in this field.",
  "My experience has prepared me to contribute from day one.",
  "I have a track record of dependable delivery and collaboration.",
  "I combine hands-on skill with clear communication.",
  "I focus on quality, ownership, and measurable outcomes.",
  "I thrive in environments that value craftsmanship and teamwork.",
  "I am known for steady execution and thoughtful problem-solving.",
  "I bring both technical ability and professional maturity.",
  "I have consistently met deadlines while maintaining high standards.",
  "I enjoy turning requirements into reliable, user-focused results.",
  "I work well with cross-functional partners and stakeholders.",
  "I am comfortable owning features from design through release.",
  "I learn quickly and adapt to new tools and processes.",
  "I take pride in writing maintainable, well-tested work.",
  "I balance speed with care when shipping to production.",
  "I contribute best when goals are clear and impact matters.",
  "I have grown through challenging projects and real accountability.",
  "I am proactive about feedback, documentation, and improvement.",
  "I stay calm under pressure and prioritize what moves the needle.",
  "I am motivated by roles where I can see the effect of my work.",
  "I have supported teams through launches, fixes, and iteration.",
  "I value transparency, respect, and steady communication.",
  "I am organized, dependable, and detail-oriented.",
  "I approach problems methodically and follow through to completion.",
  "I am ready to apply what I have learned in a new setting.",
  "I am looking for a team where I can contribute and keep growing.",
  "I bring energy and discipline to every project I touch.",
  "I have developed habits that help teams ship with confidence.",
  "I am committed to doing work I am proud to put my name on.",
  "I am prepared to make an immediate contribution.",
]

_INTRO_CLOSERS = [
  "I would welcome the opportunity to discuss my fit further.",
  "I am confident I can support your team's goals.",
  "I hope to bring my experience to your organization.",
  "I look forward to the possibility of contributing at {company}.",
  "I am ready to help {company} deliver strong results.",
  "I believe this role is the right next step for me.",
  "I am excited about what I could accomplish in this position.",
  "I would value the chance to interview and share more.",
  "I am prepared to add value in a {seniority}-level capacity.",
  "I am enthusiastic about growing with your team.",
  "I am eager to learn your processes and contribute quickly.",
  "I am confident my skills match what you are looking for.",
  "I would be glad to explain how my background maps to this role.",
  "I am motivated to do meaningful work with {company}.",
  "I am ready to take on responsibility and deliver.",
  "I hope you will consider my application favorably.",
  "I am available to discuss my experience at your convenience.",
  "I am committed to excellence in everything I deliver.",
  "I am excited to potentially join {company}.",
  "I would appreciate your consideration of my application.",
  "I am sure I can help your team move forward.",
  "I am looking forward to hearing from you.",
  "I am ready to put my skills to work for {company}.",
  "I am confident I can meet the expectations of this role.",
  "I would love to explore how I can help {company} succeed.",
]

_EXPERIENCE_FRAMES = [
  "In my work as a {title}, I have applied {skills} to deliver solid outcomes.",
  "As a {title}, I have worked hands-on with {skills} on production projects.",
  "My experience as a {title} includes practical use of {skills} in real environments.",
  "Working as a {title}, I have relied on {skills} to build and ship features.",
  "In {title} roles, I have used {skills} to support reliable product delivery.",
  "Throughout my time as a {title}, {skills} have been central to my day-to-day work.",
  "I have developed as a {title} by applying {skills} across multiple initiatives.",
  "My background as a {title} is grounded in {skills} and consistent execution.",
  "As a {title}, I have partnered with teams using {skills} to meet deadlines.",
  "I have contributed as a {title} by leveraging {skills} for maintainable solutions.",
  "In recent {title} work, {skills} enabled me to deliver with quality and speed.",
  "I have strengthened my {title} practice through repeated work with {skills}.",
  "My {title} experience reflects steady use of {skills} from planning to release.",
  "I have solved practical problems as a {title} using {skills}.",
  "I have supported users and stakeholders as a {title} through {skills}.",
  "I have grown technical breadth as a {title}, especially around {skills}.",
  "I have taken ownership as a {title} where {skills} were essential.",
  "I have collaborated effectively as a {title} while working with {skills}.",
  "I have improved processes as a {title} by applying {skills} thoughtfully.",
  "I have met business needs as a {title} with strong command of {skills}.",
  "I have shipped features as a {title} that depended on {skills}.",
  "I have debugged, refined, and released work as a {title} using {skills}.",
  "I have documented and communicated clearly as a {title} while using {skills}.",
  "I have mentored peers informally as a {title} while working in {skills}.",
  "I have balanced trade-offs as a {title} when working with {skills}.",
  "I have learned fast on the job as a {title}, particularly with {skills}.",
  "I have earned trust as a {title} through dependable work with {skills}.",
  "I have stayed current as a {title} by continuing to practice {skills}.",
  "I have contributed to releases as a {title} where {skills} were critical.",
  "I have built momentum as a {title} by delivering with {skills}.",
]

_EXPERIENCE_OUTCOMES = [
  "That work taught me to prioritize clarity, testing, and follow-through.",
  "Those projects strengthened my ability to collaborate under deadlines.",
  "I learned to communicate progress clearly and unblock teammates quickly.",
  "I focused on maintainable solutions that teams could extend later.",
  "I paid attention to edge cases, performance, and user experience.",
  "I took feedback seriously and iterated until outcomes were solid.",
  "I kept documentation current so handoffs stayed smooth.",
  "I balanced independent work with regular check-ins.",
  "I stayed organized when requirements shifted mid-sprint.",
  "I treated production issues with urgency and care.",
  "I helped reduce rework by clarifying scope early.",
  "I supported releases with careful verification.",
  "I contributed ideas that improved workflow for the team.",
  "I remained calm when priorities changed.",
  "I built trust by doing what I said I would do.",
  "I looked for small improvements that compounded over time.",
  "I aligned my work with broader product goals.",
  "I asked questions when ambiguity could slow delivery.",
  "I paired with others when it sped up results.",
  "I left codebases easier to work in than I found them.",
  "I respected code review as a tool for quality.",
  "I tracked tasks transparently so nothing slipped.",
  "I celebrated team wins and shared credit.",
  "I stayed curious about better tools and methods.",
  "I kept learning while delivering on commitments.",
  "I brought discipline to estimates and execution.",
  "I made reliability a habit, not an afterthought.",
  "I connected technical choices to user impact.",
  "I showed up prepared for standups and planning.",
  "I treated every delivery as a reflection of my standards.",
]

_SKILLS_FRAMES = [
  "{verb} {skills}, I am prepared for the core demands of this {seniority}-level {title} role.",
  "My command of {skills} aligns closely with the requirements you outlined.",
  "I am confident in {skills} and how they apply to this {title} position.",
  "The role's needs match my strengths in {skills}.",
  "I can apply {skills} effectively in a fast-moving team environment.",
  "I have practiced {skills} enough to contribute without a long ramp-up.",
  "I understand how {skills} fit into day-to-day {title} responsibilities.",
  "I am ready to put {skills} to work on meaningful tasks at {company}.",
  "I bring dependable skill in {skills} plus a willingness to learn your stack.",
  "I have used {skills} in ways that map directly to this opening.",
  "I am comfortable going deep on {skills} when the work requires it.",
  "I can collaborate using {skills} while respecting team conventions.",
  "I stay current with {skills} through practice and continuous learning.",
  "I pair strong fundamentals in {skills} with professional communication.",
  "I am organized about how I apply {skills} across a sprint.",
  "I know when to lean on {skills} and when to ask for alignment.",
  "I have built confidence in {skills} through repeated delivery.",
  "I can explain my work with {skills} clearly to non-technical partners.",
  "I treat {skills} as tools to solve problems, not ends in themselves.",
  "I am motivated to deepen my use of {skills} in this role.",
  "I have a practical, results-oriented approach to {skills}.",
  "I am detail-minded when working with {skills}.",
  "I can juggle multiple priorities while applying {skills} carefully.",
  "I am adaptable if {company} uses {skills} in a specific way.",
  "I am ready to demonstrate my ability with {skills} in interview and on the job.",
  "I connect {skills} to outcomes users and stakeholders care about.",
  "I am proud of the standard I hold when working with {skills}.",
  "I am eager to expand how I use {skills} on your team.",
  "I believe {skills} are a strong foundation for success in this role.",
  "I am prepared to grow my impact through {skills} at {company}.",
]

_SKILLS_VERBS = [
  "Through", "With", "Using", "By applying", "Drawing on",
  "Building on", "Leaning on", "Grounded in", "Rooted in", "Supported by",
  "Backed by", "Informed by", "Strengthened by", "Guided by", "Powered by",
  "Centered on", "Focused on", "Anchored in", "Developed through", "Honed through",
  "Refined through", "Demonstrated through", "Validated through", "Proven through",
  "Established through", "Cultivated through", "Sharpened through", "Expanded through",
  "Deepened through", "Practiced through",
]

_COMPANY_FRAMES = [
  "I am especially interested in {company} because of the impact a {title} can have here.",
  "Joining {company} appeals to me as a place to do focused, high-quality work.",
  "I am motivated by the opportunity to contribute to {company}'s goals as a {title}.",
  "What draws me to {company} is the chance to apply my skills where they matter.",
  "I would value growing with {company} while supporting real product outcomes.",
  "I see {company} as a strong fit for the next step in my career as a {title}.",
  "I am excited about the possibility of building with the team at {company}.",
  "I believe {company} offers the kind of work environment where I do my best.",
  "I am eager to learn how {company} operates and add value quickly.",
  "I respect the work coming out of {company} and want to be part of it.",
  "I am looking for a team like {company}'s where craftsmanship is taken seriously.",
  "I am confident I can support {company}'s mission through steady delivery.",
  "I would be proud to represent {company} in a {title} capacity.",
  "I am attracted to the problems {company} is solving.",
  "I want to contribute to {company}'s momentum in this role.",
  "I am ready to align my efforts with {company}'s priorities.",
  "I am interested in how {company} serves its users and stakeholders.",
  "I would welcome the accountability that comes with joining {company}.",
  "I am prepared to collaborate openly within {company}'s culture.",
  "I am enthusiastic about bringing my experience to {company}.",
  "I hope to make a visible difference at {company} as a {title}.",
  "I am committed to the standard of work I would bring to {company}.",
  "I am optimistic about what I could learn and deliver at {company}.",
  "I am drawn to teams that care about execution, and {company} feels like one.",
  "I would treat an offer from {company} as a serious opportunity to contribute.",
  "I am ready to invest myself in {company}'s success.",
  "I am interested in long-term growth with {company}.",
  "I believe my work ethic would fit well at {company}.",
  "I am excited to potentially call {company} my next professional home.",
  "I would approach this opportunity at {company} with focus and gratitude.",
]

_COMPANY_REASONS = [
  "The role itself is a natural match for my background.",
  "I appreciate organizations that reward reliability and initiative.",
  "I am looking for meaningful problems and clear ownership.",
  "I want to work where quality is noticed and expected.",
  "I am ready for a team that moves with purpose.",
  "I value transparency, respect, and steady communication.",
  "I am motivated by roles with visible impact.",
  "I am seeking a culture where I can keep learning.",
  "I am drawn to teams that ship and iterate thoughtfully.",
  "I want to contribute somewhere I can grow over time.",
  "I am interested in collaborative environments with high standards.",
  "I am ready to take on responsibility and deliver.",
  "I am excited by the chance to do work I can stand behind.",
  "I am looking for alignment between my skills and your needs.",
  "I am prepared to engage fully from the start.",
  "I am optimistic about building something useful together.",
  "I am focused on finding the right long-term fit.",
  "I am eager to bring discipline and curiosity to the team.",
  "I am motivated by teams that care about users.",
  "I am ready to match my pace to your priorities.",
  "I am interested in contributing beyond my immediate tasks.",
  "I am hopeful this is where I can do my best work.",
  "I am committed to professionalism in every interaction.",
  "I am excited to potentially grow with you.",
  "I am looking forward to learning your ways of working.",
  "I am confident I would represent your team well.",
  "I am ready to earn trust through consistent delivery.",
  "I am enthusiastic about this specific opening.",
  "I am prepared to interview and discuss fit in detail.",
  "I am grateful for your consideration of my application.",
]

_CLOSING_LINES = [
  "Thank you for your time and consideration.",
  "Thank you for reviewing my application.",
  "I appreciate your consideration of my candidacy.",
  "Thank you for the opportunity to apply.",
  "I am grateful for your time and attention.",
  "Thank you for reading my application.",
  "I appreciate the chance to be considered.",
  "Thank you for evaluating my fit for this role.",
  "I value your time and thoughtful review.",
  "Thank you for your consideration and time.",
  "I appreciate your attention to my application.",
  "Thank you for taking the time to review my materials.",
  "I am thankful for your consideration.",
  "Thank you for your time today.",
  "I appreciate your openness to new applicants.",
  "Thank you for considering my background.",
  "I am grateful for this opportunity to apply.",
  "Thank you for your patience and consideration.",
  "I appreciate your review and hope to connect soon.",
  "Thank you for your thoughtful consideration.",
  "I am thankful you took the time to read this.",
  "Thank you for your time and professional courtesy.",
  "I appreciate your consideration of my experience.",
  "Thank you for reviewing my letter and resume.",
  "I am grateful for the chance to introduce myself.",
  "Thank you for your attention to my application.",
  "I appreciate your time and hope to speak soon.",
  "Thank you for giving my application your consideration.",
  "I am thankful for the opportunity to be considered.",
  "Thank you for your time and interest.",
]

_CLOSING_FOLLOWUPS = [
  "I look forward to hearing from you.",
  "I hope to discuss my application with you soon.",
  "I would welcome a conversation about this role.",
  "I am available to interview at your convenience.",
  "I would be glad to provide any additional information.",
  "I hope we can speak about how I can contribute.",
  "I look forward to the possibility of an interview.",
  "I am happy to meet and share more about my experience.",
  "I would appreciate the chance to discuss fit in person or by call.",
  "I am eager to learn more about the team and role.",
  "I hope to connect with you in the near future.",
  "I am ready to discuss next steps whenever works for you.",
  "I would value a few minutes to introduce myself further.",
  "I am confident a conversation would be worthwhile.",
  "I look forward to your response.",
  "I am excited about the possibility of moving forward.",
  "I hope to join you in conversation soon.",
  "I am prepared to answer any questions you may have.",
  "I would welcome feedback on my application.",
  "I am enthusiastic about potential next steps.",
  "I hope to demonstrate my fit in an interview.",
  "I am available on short notice if helpful.",
  "I would be honored to discuss this opportunity further.",
  "I look forward to contributing if selected.",
  "I am ready to proceed whenever you are.",
  "I hope my application merits a closer look.",
  "I am excited to explore this fit with your team.",
  "I would appreciate the opportunity to elaborate.",
  "I am hopeful we can connect soon.",
  "I look forward to the next step in your process.",
]

# --- Tone overlays (multiply variation per tone) --------------------------------

_TONE_OVERLAYS: dict[str, dict[str, list[str]]] = {
  "professional": {
    "intro_adj": [
      "I am confident", "I am prepared", "I am well-positioned", "I am ready",
      "I am qualified", "I am equipped", "I am suited", "I am aligned",
    ],
    "exp_adj": ["consistently", "reliably", "effectively", "professionally", "diligently"],
    "skills_adj": ["solid", "proven", "practical", "relevant", "applicable"],
    "company_adj": ["particularly", "especially", "genuinely", "sincerely", "clearly"],
    "closing_adj": ["Sincerely", "Respectfully", "Best regards", "Kind regards", "Warm regards"],
  },
  "casual": {
    "intro_adj": [
      "I'd love", "I'm keen", "I'm happy", "I'm excited", "I'm ready",
      "I'm pumped", "I'm stoked", "I'm game", "I'm all in", "I'm up for it",
    ],
    "exp_adj": ["regularly", "often", "day to day", "on the job", "in practice"],
    "skills_adj": ["solid", "handy", "useful", "real-world", "everyday"],
    "company_adj": ["really", "honestly", "genuinely", "truly", "definitely"],
    "closing_adj": ["Thanks", "Cheers", "Best", "Take care", "All the best"],
  },
  "friendly": {
    "intro_adj": [
      "I'm delighted", "I'm pleased", "I'm happy", "I'm glad", "I'm excited",
      "I'm enthusiastic", "I'm hopeful", "I'm eager", "I'm looking forward", "I'm thrilled",
    ],
    "exp_adj": ["warmly", "collaboratively", "supportively", "kindly", "openly"],
    "skills_adj": ["helpful", "welcoming", "team-friendly", "approachable", "supportive"],
    "company_adj": ["warmly", "happily", "gladly", "cheerfully", "openly"],
    "closing_adj": ["Warmly", "With appreciation", "Kindly", "With thanks", "Friendly regards"],
  },
  "formal": {
    "intro_adj": [
      "I respectfully submit", "I hereby express", "I formally present", "I wish to convey",
      "I take pleasure in", "I have the honor to", "I am pleased to", "I am privileged to",
    ],
    "exp_adj": ["methodically", "systematically", "rigorously", "precisely", "formally"],
    "skills_adj": ["demonstrated", "established", "documented", "verified", "recognized"],
    "company_adj": ["respectfully", "formally", "duly", "properly", "appropriately"],
    "closing_adj": ["Respectfully yours", "Yours faithfully", "Yours sincerely", "With esteem", "Cordially"],
  },
  "confident": {
    "intro_adj": [
      "I am certain", "I am convinced", "I am assured", "I am confident", "I am positive",
      "I know", "I am sure", "I am ready", "I am prepared to excel", "I am built for",
    ],
    "exp_adj": ["decisively", "assertively", "boldly", "directly", "forcefully"],
    "skills_adj": ["strong", "sharp", "commanding", "decisive", "impactful"],
    "company_adj": ["firmly", "decisively", "clearly", "directly", "strongly"],
    "closing_adj": ["Confidently", "With conviction", "Assuredly", "Decisively", "Boldly"],
  },
  "enthusiastic": {
    "intro_adj": [
      "I am thrilled", "I am energized", "I am fired up", "I am passionate", "I am animated",
      "I am buzzing", "I am eager", "I am excited", "I am motivated", "I am inspired",
    ],
    "exp_adj": ["energetically", "passionately", "vibrantly", "actively", "dynamically"],
    "skills_adj": ["energizing", "exciting", "dynamic", "vibrant", "motivating"],
    "company_adj": ["excitedly", "passionately", "energetically", "eagerly", "vividly"],
    "closing_adj": ["Enthusiastically", "With excitement", "Eagerly", "With energy", "Passionately"],
  },
  "persuasive": {
    "intro_adj": [
      "I am compelled to apply because", "I am convinced I can", "I am certain I will",
      "I am ready to prove", "I am prepared to demonstrate", "I am positioned to deliver",
      "I am equipped to drive", "I am set to contribute", "I am geared to help", "I am aimed at",
    ],
    "exp_adj": ["strategically", "persuasively", "compellingly", "convincingly", "impactfully"],
    "skills_adj": ["compelling", "strategic", "high-impact", "results-driven", "value-adding"],
    "company_adj": ["strategically", "compellingly", "convincingly", "decisively", "purposefully"],
    "closing_adj": ["Persuasively", "With conviction", "Strategically", "Purposefully", "Compellingly"],
  },
  "neutral": {
    "intro_adj": [
      "I am applying", "I am submitting", "I am presenting", "I am offering", "I am putting forward",
      "I am forwarding", "I am sharing", "I am providing", "I am supplying", "I am sending",
    ],
    "exp_adj": ["typically", "generally", "usually", "commonly", "routinely"],
    "skills_adj": ["adequate", "sufficient", "appropriate", "suitable", "fitting"],
    "company_adj": ["simply", "directly", "plainly", "matter-of-factly", "clearly"],
    "closing_adj": ["Regards", "Sincerely", "Thank you", "Best", "Respectfully"],
  },
}

_YEARS_PHRASES = [
  " With {years}+ years of experience,",
  " Having spent {years}+ years in the field,",
  " After {years}+ years of hands-on work,",
  " With more than {years} years of practice,",
  " Across {years}+ years of professional experience,",
  " Building on {years}+ years of experience,",
  " Drawing from {years}+ years in similar roles,",
  " Leveraging {years}+ years of background,",
]


def _normalize_tone(tone: str | None) -> str:
  t = (tone or "professional").strip().lower()
  return t if t in VALID_TONES else "professional"


def _pick(pool: list[str], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _combo_parts(seed: int, pools: list[list[str]]) -> list[str]:
  parts: list[str] = []
  s = max(0, int(seed))
  for pool in pools:
    if not pool:
      parts.append("")
      continue
    parts.append(pool[s % len(pool)])
    s //= max(len(pool), 1)
  return parts


def template_combination_counts() -> dict[str, int]:
  """Approximate unique combinations per section (x8 tones for overlay slot)."""
  tone_mult = len(VALID_TONES)
  return {
    "greeting": len(_GREETING_STEMS) * tone_mult,
    "introduction": len(_INTRO_OPENERS) * len(_INTRO_BRIDGES) * len(_INTRO_CLOSERS) * len(_YEARS_PHRASES) * tone_mult,
    "experience": len(_EXPERIENCE_FRAMES) * len(_EXPERIENCE_OUTCOMES) * tone_mult,
    "skills": len(_SKILLS_FRAMES) * len(_SKILLS_VERBS) * tone_mult,
    "company": len(_COMPANY_FRAMES) * len(_COMPANY_REASONS) * tone_mult,
    "closing": len(_CLOSING_LINES) * len(_CLOSING_FOLLOWUPS) * tone_mult,
  }


def _tone_banks(tone: str) -> dict[str, list[str]]:
  return _TONE_OVERLAYS.get(tone, _TONE_OVERLAYS["professional"])


def _fmt(template: str, **kwargs: Any) -> str:
  try:
    return template.format(**kwargs)
  except KeyError:
    return template


def generate_greeting(
  company: str,
  applicant_name: str | None,
  seed: int,
  tone: str,
) -> str:
  tone = _normalize_tone(tone)
  banks = _tone_banks(tone)
  stem = _pick(_GREETING_STEMS, seed)
  greeting = _fmt(stem, company=company)
  if applicant_name and seed % 7 == 0:
    return f"Dear {company} Team,"
  _ = banks  # tone reserved for future greeting variants
  return greeting


def generate_introduction(
  title: str,
  company: str,
  years: int,
  seniority: str,
  seed: int,
  tone: str,
) -> str:
  tone = _normalize_tone(tone)
  tone_seed = seed + sum(ord(c) for c in tone) * 17
  opener, bridge, closer = _combo_parts(tone_seed, [_INTRO_OPENERS, _INTRO_BRIDGES, _INTRO_CLOSERS])
  opener = _fmt(opener, title=title, company=company)
  closer = _fmt(closer, company=company, seniority=seniority)
  parts = [opener]
  if years > 0:
    parts.append(_fmt(_pick(_YEARS_PHRASES, tone_seed + 3), years=years).strip().rstrip(","))
  parts.append(bridge)
  parts.append(closer)
  return " ".join(p.strip() for p in parts if p.strip())


def generate_experience_paragraph(
  title: str,
  skills: str,
  highlight: str | None,
  seed: int,
  tone: str,
) -> str:
  tone = _normalize_tone(tone)
  tone_seed = seed + sum(ord(c) for c in tone) * 23
  frame, outcome = _combo_parts(tone_seed + 17, [_EXPERIENCE_FRAMES, _EXPERIENCE_OUTCOMES])
  frame = _fmt(frame, title=title, skills=skills)
  if highlight and len(highlight) > 30:
    body = re.sub(r"^(?:I am|I'm|I have|I've)\s+", "", highlight, flags=re.I)
    return f"{frame} {body.rstrip('.')}. {outcome}"
  return f"{frame} {outcome}"


def generate_skills_paragraph(
  skills: str,
  title: str,
  company: str,
  seniority: str,
  seed: int,
  tone: str,
) -> str:
  tone = _normalize_tone(tone)
  tone_seed = seed + sum(ord(c) for c in tone) * 29
  frame, verb = _combo_parts(tone_seed + 23, [_SKILLS_FRAMES, _SKILLS_VERBS])
  return _fmt(frame, verb=verb, skills=skills, title=title, company=company, seniority=seniority)


def generate_company_paragraph(
  company: str,
  title: str,
  seed: int,
  tone: str,
) -> str:
  tone = _normalize_tone(tone)
  tone_seed = seed + sum(ord(c) for c in tone) * 31
  frame, reason = _combo_parts(tone_seed + 31, [_COMPANY_FRAMES, _COMPANY_REASONS])
  frame = _fmt(frame, company=company, title=title)
  return f"{frame} {reason}"


def generate_closing(seed: int, tone: str) -> str:
  tone = _normalize_tone(tone)
  line, follow = _combo_parts(seed + 41, [_CLOSING_LINES, _CLOSING_FOLLOWUPS])
  return f"{line} {follow}"


def generate_signature(applicant_name: str | None, seed: int, tone: str) -> str:
  tone = _normalize_tone(tone)
  sign = _pick(_tone_banks(tone)["closing_adj"], seed + 43)
  name = applicant_name or "[Your Name]"
  return f"{sign},\n{name}"
