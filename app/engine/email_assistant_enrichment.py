"""Email Assistant — analysis, generation, and quality stages (production v5.2)."""

from __future__ import annotations

import re
from typing import Any

from app.engine.seo_retrieval_engine import detect_nsfw_topic

GENERATOR_VERSION = "email-assistant-v5.2"

ARCHITECTURE_FLOW = [
  "input",
  "input_validator",
  "content_policy_gate",
  "language_detector",
  "email_type_classifier",
  "intent_detector",
  "recipient_analyzer",
  "relationship_classifier",
  "context_extractor",
  "thread_parser",
  "entity_extraction",
  "sentiment_detection",
  "urgency_detector",
  "formality_detector",
  "tone_optimizer",
  "culture_locale_adapter",
  "business_domain_classifier",
  "email_structure_planner",
  "subject_generator",
  "opening_generator",
  "body_generator",
  "cta_generator",
  "closing_generator",
  "signature_generator",
  "grammar_checker",
  "style_optimizer",
  "readability_analyzer",
  "spam_score_checker",
  "pii_security_filter",
  "professionalism_validator",
  "quality_scorer",
  "alternative_versions",
  "final_output",
]

VALID_TONES = frozenset({"professional", "casual", "friendly", "formal"})

SUBJECT_MIN = 25
SUBJECT_MAX = 60

_EMAIL_TYPE_HINTS: dict[str, tuple[str, ...]] = {
  "sales": ("proposal", "demo", "pricing", "offer", "partnership", "cold", "quote"),
  "marketing": ("newsletter", "campaign", "promotion", "launch", "webinar"),
  "recruitment": ("interview", "job", "application", "resume", "hiring", "recruiter", "candidate"),
  "support": ("support", "issue", "ticket", "help", "bug", "problem", "error"),
  "meeting": ("meeting", "schedule", "calendar", "call", "sync", "agenda"),
  "thank_you": ("thank", "thanks", "grateful", "appreciation"),
  "apology": ("sorry", "apolog", "regret", "mistake", "inconvenience"),
  "follow_up": ("follow up", "follow-up", "checking in", "reminder", "circling back"),
  "invoice": ("invoice", "payment", "billing", "quote", "receipt"),
  "complaint": ("complaint", "unhappy", "dissatisfied", "refund", "disappointed"),
  "networking": ("connect", "introduction", "network", "referral"),
  "proposal": ("proposal", "quotation", "rfp", "scope of work"),
  "onboarding": ("onboarding", "welcome aboard", "getting started"),
  "project_update": ("project update", "status update", "milestone", "deliverable"),
}

_INTENT_HINTS: dict[str, tuple[str, ...]] = {
  "request": ("please", "could you", "would you", "request", "need", "ask"),
  "inform": ("update", "inform", "sharing", "fyi", "notice", "heads up"),
  "schedule": ("schedule", "meeting", "call", "book", "calendar", "availability"),
  "sell": ("offer", "solution", "value", "benefit", "demo", "roi"),
  "thank": ("thank", "thanks", "grateful", "appreciate"),
  "apologize": ("sorry", "apolog", "regret"),
  "follow_up": ("follow up", "following up", "check in", "touch base"),
  "confirm": ("confirm", "confirmation", "approved", "acknowledge"),
  "support": ("help", "issue", "problem", "support", "resolve"),
  "negotiate": ("negotiate", "terms", "counter", "proposal"),
}

_RECIPIENT_HINTS: dict[str, tuple[str, ...]] = {
  "CEO": ("ceo", "chief executive", "founder", "president"),
  "HR": ("hr", "human resources", "recruiter", "talent", "hiring manager"),
  "Manager": ("manager", "director", "head of", "vp", "lead"),
  "Customer": ("customer", "client", "buyer"),
  "Vendor": ("vendor", "supplier", "partner"),
  "Finance": ("finance", "accounts payable", "billing"),
}

_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
  "Technology": ("software", "saas", "app", "ai", "cloud", "tech", "developer", "api"),
  "Healthcare": ("health", "medical", "hospital", "patient", "clinic"),
  "Finance": ("finance", "bank", "investment", "insurance", "fintech"),
  "Education": ("school", "university", "education", "course", "student"),
  "Legal": ("legal", "law", "attorney", "compliance"),
  "Retail": ("retail", "store", "ecommerce", "shop", "consumer"),
  "Real Estate": ("property", "real estate", "apartment", "rental"),
  "Travel": ("travel", "hotel", "flight", "hospitality"),
  "Manufacturing": ("manufacturing", "factory", "supply chain", "logistics"),
  "Consulting": ("consulting", "advisory", "professional services"),
}

_SPAM_WORDS = (
  "free!!!", "buy now", "limited offer", "click here", "act now", "winner",
  "100% guaranteed", "no obligation", "risk free", "earn money fast",
  "dear friend", "congratulations", "you have won",
)

_PII_PATTERNS = (
  (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
  (r"\b(?:\d[ -]*?){13,19}\b", "credit_card"),
  (r"\bpassword\s*[:=]\s*\S+", "password"),
  (r"\bsk-[a-zA-Z0-9]{20,}\b", "api_key"),
  (r"\bOTP\s*[:=]?\s*\d{4,8}\b", "otp"),
)

_GREETINGS: dict[str, list[str]] = {
  "professional": ["Hello,", "Good day,", "Hi there,"],
  "formal": ["Dear Team,", "Dear Sir or Madam,", "To Whom It May Concern,"],
  "friendly": ["Hi,", "Hello there,", "Hope you're doing well —"],
  "casual": ["Hey,", "Hi!", "Hello —"],
}

_CLOSINGS: dict[str, list[str]] = {
  "professional": ["Best regards,", "Kind regards,", "Thank you,"],
  "formal": ["Sincerely,", "Yours faithfully,", "Respectfully,"],
  "friendly": ["Warm regards,", "Thanks so much,", "Best,"],
  "casual": ["Cheers,", "Thanks,", "Talk soon,"],
}

_CTA_BY_INTENT: dict[str, list[str]] = {
  "schedule": [
    "Would you be open to a brief call this week?",
    "Please share a few times that work for you.",
  ],
  "sell": [
    "Would you be open to a 15-minute conversation to explore this further?",
    "I'd welcome the chance to walk you through how this could help.",
  ],
  "request": [
    "I'd appreciate your thoughts when you have a moment.",
    "Please let me know if this works on your end.",
  ],
  "inform": [
    "Please let me know if you have any questions.",
    "Happy to clarify anything if needed.",
  ],
  "follow_up": [
    "I'd appreciate a quick update when convenient.",
    "Looking forward to hearing from you.",
  ],
  "thank": ["Thanks again for your time and support.", ""],
  "support": [
    "Please let me know if you need any additional details.",
    "I'm happy to help further if needed.",
  ],
  "apologize": [
    "Thank you for your patience and understanding.",
    "Please let me know if there's anything else I can do to help.",
  ],
  "confirm": ["Please confirm receipt at your earliest convenience.", ""],
}

# Per-industry cold-email framing (opener context, value intro, domain CTA).
_COLD_DOMAIN_TEMPLATES: dict[str, dict[str, str]] = {
  "Technology": {
    "context_line": "We partner with product and engineering teams on {purpose}.",
    "value_intro": "Typical outcomes for tech teams:",
    "cta": "Would a 15-minute technical overview be useful this week?",
  },
  "Healthcare": {
    "context_line": "We work with healthcare organizations focused on {purpose}.",
    "value_intro": "What peers in healthcare often see:",
    "cta": "Could we schedule a brief call to discuss fit for your organization?",
  },
  "Finance": {
    "context_line": "We support finance teams navigating {purpose}.",
    "value_intro": "Results we commonly deliver:",
    "cta": "Would you be open to a short conversation about your current priorities?",
  },
  "Education": {
    "context_line": "We help education teams improve {purpose}.",
    "value_intro": "Impact for institutions like yours:",
    "cta": "Would a quick call to explore alignment make sense?",
  },
  "Legal": {
    "context_line": "We assist legal and compliance teams with {purpose}.",
    "value_intro": "Key benefits:",
    "cta": "May I share a concise overview at your convenience?",
  },
  "Retail": {
    "context_line": "We help retail and e-commerce brands with {purpose}.",
    "value_intro": "What similar brands achieve:",
    "cta": "Would you be open to a brief call to see if this fits your roadmap?",
  },
  "Real Estate": {
    "context_line": "We support property teams working on {purpose}.",
    "value_intro": "Outcomes for real estate operators:",
    "cta": "Could we connect for a short intro call this week?",
  },
  "Travel": {
    "context_line": "We partner with travel and hospitality teams on {purpose}.",
    "value_intro": "Results in this sector:",
    "cta": "Would a 15-minute intro call work for you?",
  },
  "Manufacturing": {
    "context_line": "We help manufacturing and supply-chain teams with {purpose}.",
    "value_intro": "Operational improvements we deliver:",
    "cta": "Would you be open to a brief discussion about your current challenges?",
  },
  "Consulting": {
    "context_line": "We collaborate with advisory firms on {purpose}.",
    "value_intro": "Value for consulting teams:",
    "cta": "Could we schedule a short exploratory call?",
  },
  "General Business": {
    "context_line": "We help teams like yours with {purpose}.",
    "value_intro": "What we offer:",
    "cta": "Would you be open to a brief conversation to explore this?",
  },
}

# Localized greetings/closings when input language is detected as non-English.
_LOCALE_PHRASES: dict[str, dict[str, list[str]]] = {
  "es": {
    "professional": {"greetings": ["Hola,", "Buenos días,"], "closings": ["Saludos cordiales,", "Atentamente,"]},
    "formal": {"greetings": ["Estimado equipo,", "Muy señores míos:"], "closings": ["Atentamente,", "Cordialmente,"]},
    "friendly": {"greetings": ["Hola,", "¡Espero que estés bien!"], "closings": ["Un saludo,", "Gracias,"]},
    "casual": {"greetings": ["Hola,", "¡Qué tal!"], "closings": ["Saludos,", "Gracias,"]},
  },
  "fr": {
    "professional": {"greetings": ["Bonjour,", "Bonjour à vous,"], "closings": ["Cordialement,", "Bien à vous,"]},
    "formal": {"greetings": ["Madame, Monsieur,", "À l'attention de l'équipe,"], "closings": ["Je vous prie d'agréer mes salutations distinguées,", "Sincèrement,"]},
    "friendly": {"greetings": ["Bonjour,", "J'espère que vous allez bien —"], "closings": ["Bien cordialement,", "Merci,"]},
    "casual": {"greetings": ["Salut,", "Bonjour !"], "closings": ["À bientôt,", "Merci,"]},
  },
  "de": {
    "professional": {"greetings": ["Guten Tag,", "Hallo,"], "closings": ["Mit freundlichen Grüßen,", "Vielen Dank,"]},
    "formal": {"greetings": ["Sehr geehrte Damen und Herren,", "Sehr geehrte Damen und Herren,"], "closings": ["Hochachtungsvoll,", "Mit freundlichen Grüßen,"]},
    "friendly": {"greetings": ["Hallo,", "Ich hoffe, es geht Ihnen gut —"], "closings": ["Herzliche Grüße,", "Vielen Dank,"]},
    "casual": {"greetings": ["Hi,", "Hallo!"], "closings": ["Viele Grüße,", "Danke,"]},
  },
  "hi": {
    "professional": {"greetings": ["नमस्ते,", "प्रणाम,"], "closings": ["सादर,", "धन्यवाद,"]},
    "formal": {"greetings": ["माननीय महोदय/महोदया,", "प्रिय टीम,"], "closings": ["भवदीय,", "आपका आभारी,"]},
    "friendly": {"greetings": ["नमस्ते,", "आशा है आप स्वस्थ हैं —"], "closings": ["शुभकामनाएँ,", "धन्यवाद,"]},
    "casual": {"greetings": ["हाय,", "नमस्ते!"], "closings": ["शुभ रहे,", "धन्यवाद,"]},
  },
  "ar": {
    "professional": {"greetings": ["مرحباً،", "السلام عليكم،"], "closings": ["مع أطيب التحيات،", "شكراً لكم،"]},
    "formal": {"greetings": ["السادة الكرام،", "تحية طيبة،"], "closings": ["وتفضلوا بقبول فائق الاحترام،", "مع خالص التقدير،"]},
    "friendly": {"greetings": ["مرحباً،", "أتمنى أن تكونوا بخير —"], "closings": ["أطيب التحيات،", "شكراً،"]},
    "casual": {"greetings": ["أهلاً،", "مرحباً!"], "closings": ["تحياتي،", "شكراً،"]},
  },
}


def _clean(text: str | None) -> str:
  return re.sub(r"\s+", " ", (text or "").strip())


def _words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text))


def normalize_tone(tone: str | None) -> str:
  if not tone:
    return "professional"
  t = tone.strip().lower()
  return t if t in VALID_TONES else "professional"


def check_content_policy(text: str) -> dict[str, Any]:
  """Block restricted / adult-service email generation."""
  nsfw = detect_nsfw_topic(text, [])
  profile = nsfw.get("profile", "general")
  blocked = profile in ("adult_services",) or nsfw.get("is_adult", False)
  return {
    "allowed": not blocked,
    "profile": profile,
    "reason": "restricted_content" if blocked else None,
    "message": (
      "This topic cannot be used for email generation. Please use professional business content."
      if blocked else None
    ),
  }


def validate_input(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
  issues: list[str] = []
  if mode == "new_email":
    if not _clean(payload.get("context")):
      issues.append("context_required")
  elif mode == "reply":
    if not _clean(payload.get("original_email")):
      issues.append("original_email_required")
  elif mode == "cold_email":
    for key in ("company_name", "purpose_offer", "value_proposition"):
      if not _clean(payload.get(key)):
        issues.append(f"{key}_required")
  else:
    issues.append("invalid_mode")
  if payload.get("tone") and normalize_tone(payload.get("tone")) not in VALID_TONES:
    issues.append("invalid_tone")
  return {"valid": not issues, "issues": issues, "mode": mode}


def parse_key_points(text: str) -> list[str]:
  """Split context/key points into clean bullet lines."""
  raw = (text or "").strip()
  if not raw:
    return []
  chunks = re.split(r"[\n\r]+|(?:\s*;\s+)|(?:\s*•\s+)", raw)
  points: list[str] = []
  for chunk in chunks:
    chunk = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", chunk.strip())
    chunk = _clean(chunk)
    if len(chunk) < 3:
      continue
    if len(chunk) > 220:
      chunk = chunk[:217].rsplit(" ", 1)[0] + "..."
    points.append(chunk)
  if not points and raw:
    for part in re.split(r",\s*(?=[A-Z])", raw):
      part = _clean(part)
      if len(part) > 3:
        points.append(part)
  return points[:8]


def parse_original_email(text: str) -> dict[str, Any]:
  """Extract subject line and body from pasted email thread."""
  raw = (text or "").strip()
  subject = ""
  body = raw
  m = re.search(r"^subject\s*:\s*(.+)$", raw, re.I | re.M)
  if m:
    subject = _clean(m.group(1))[:120]
    body = raw[m.end():].strip()
  from_m = re.search(r"^from\s*:\s*(.+)$", raw, re.I | re.M)
  sender = _clean(from_m.group(1))[:80] if from_m else ""
  preview = _clean(body)[:280]
  return {
    "subject": subject,
    "sender": sender,
    "body_preview": preview,
    "body": body,
    "has_thread": bool(subject or sender),
  }


def detect_language(text: str) -> dict[str, Any]:
  low = text.lower()
  hints = {
    "en": ("the", "and", "please", "thank", "regards", "hello", "dear"),
    "es": ("hola", "gracias", "estimado", "saludos", "buenos"),
    "fr": ("bonjour", "merci", "cordialement", "madame"),
    "de": ("hallo", "danke", "freundliche", "guten"),
    "hi": ("नमस्ते", "धन्यवाद", "आपका", "सादर"),
    "ar": ("مرحبا", "شكرا", "تحية"),
    "pt": ("obrigado", "prezado", "atenciosamente"),
    "zh": ("您好", "谢谢", "此致"),
  }
  scores = {k: sum(1 for h in v if h in low) for k, v in hints.items()}
  best = max(scores, key=scores.get) if scores else "en"
  if scores.get(best, 0) == 0:
    best = "en"
  labels = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "hi": "Hindi", "ar": "Arabic", "pt": "Portuguese", "zh": "Chinese",
  }
  return {"language": labels.get(best, "English"), "bcp47": best, "source": "auto_detect"}


def classify_email_subtype(mode: str, text: str) -> dict[str, Any]:
  hay = text.lower()
  if mode == "cold_email":
    return {"primary_type": "sales", "subtype": "cold_outreach", "mode": mode}
  if mode == "reply":
    return {"primary_type": "reply", "subtype": "response", "mode": mode}
  matched: list[str] = []
  for etype, hints in _EMAIL_TYPE_HINTS.items():
    if any(h in hay for h in hints):
      matched.append(etype)
  primary = matched[0] if matched else "business"
  return {"primary_type": primary, "subtypes": matched[:5], "mode": mode}


def detect_intent(text: str, mode: str) -> dict[str, Any]:
  hay = text.lower()
  scores: dict[str, int] = {}
  for intent, hints in _INTENT_HINTS.items():
    scores[intent] = sum(1 for h in hints if h in hay)
  if mode == "cold_email":
    scores["sell"] = scores.get("sell", 0) + 3
  if mode == "reply":
    scores["inform"] = scores.get("inform", 0) + 1
    if any(w in hay for w in ("sorry", "apolog", "issue", "problem")):
      scores["apologize"] = scores.get("apologize", 0) + 2
      scores["support"] = scores.get("support", 0) + 1
  primary = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else "inform"
  return {"primary_intent": primary, "scores": scores}


def analyze_recipient(text: str) -> dict[str, Any]:
  hay = text.lower()
  roles: list[str] = []
  for role, hints in _RECIPIENT_HINTS.items():
    if any(h in hay for h in hints):
      roles.append(role)
  return {"roles": roles or ["Unknown"], "primary_role": roles[0] if roles else "Unknown"}


def classify_relationship(mode: str, text: str) -> dict[str, Any]:
  if mode == "cold_email":
    return {"relationship": "cold_prospect", "warmth": "cold"}
  if mode == "reply":
    return {"relationship": "existing_thread", "warmth": "warm"}
  return {"relationship": "professional_contact", "warmth": "neutral"}


def extract_context(mode: str, payload: dict[str, Any]) -> dict[str, Any]:
  if mode == "new_email":
    points = parse_key_points(payload.get("context", ""))
    subject = _clean(payload.get("subject")) or "Quick update"
    return {
      "subject": subject,
      "key_points": points,
      "key_points_raw": _clean(payload.get("context")),
      "summary": "; ".join(points[:4]) or _clean(payload.get("context"))[:300],
    }
  if mode == "reply":
    original = _clean(payload.get("original_email"))
    thread = parse_original_email(original)
    points = parse_key_points(payload.get("reply_points", ""))
    return {
      "original_email": original,
      "thread": thread,
      "reply_points": points,
      "reply_points_raw": _clean(payload.get("reply_points")),
      "summary": thread.get("body_preview") or original[:300],
    }
  company = _clean(payload.get("company_name"))
  return {
    "company_name": company,
    "purpose_offer": _clean(payload.get("purpose_offer")),
    "value_proposition": _clean(payload.get("value_proposition")),
    "summary": f"{company}: {_clean(payload.get('value_proposition'))[:200]}",
  }


def extract_entities(text: str) -> list[dict[str, str]]:
  entities: list[dict[str, str]] = []
  seen: set[str] = set()
  for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", text):
    val = m.group(0)
    if val.lower() not in seen:
      seen.add(val.lower())
      entities.append({"type": "name_or_org", "value": val})
  for m in re.finditer(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", text):
    entities.append({"type": "email", "value": m.group(0)})
  for m in re.finditer(
    r"\b(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[^\n,]{0,24}\b",
    text,
    re.I,
  ):
    entities.append({"type": "date", "value": m.group(0)[:40]})
  for m in re.finditer(r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d{2})?|\d+%)", text):
    entities.append({"type": "currency_or_metric", "value": m.group(0)})
  for m in re.finditer(r"\b(?:INV|PO|ORD|TKT)[-#]?\w+\b", text, re.I):
    entities.append({"type": "reference_id", "value": m.group(0)})
  return entities[:24]


def detect_sentiment(text: str) -> dict[str, Any]:
  low = text.lower()
  neg = sum(1 for w in ("sorry", "unfortunately", "issue", "problem", "concern", "urgent", "frustrat", "disappoint") if w in low)
  pos = sum(1 for w in ("thank", "great", "excited", "happy", "pleased", "appreciate", "glad") if w in low)
  if neg > pos + 1:
    label = "concerned" if "urgent" in low else "neutral_negative"
  elif pos > neg:
    label = "positive"
  else:
    label = "neutral"
  return {"sentiment": label, "positive_signals": pos, "negative_signals": neg}


def detect_urgency(text: str) -> dict[str, Any]:
  low = text.lower()
  if any(w in low for w in ("asap", "urgent", "immediately", "today", "deadline", "critical", "eod")):
    return {"level": "high"}
  if any(w in low for w in ("soon", "this week", "follow up", "reminder", "tomorrow")):
    return {"level": "medium"}
  return {"level": "low"}


def detect_formality(tone: str, text: str) -> dict[str, Any]:
  mapping = {
    "formal": "very_formal",
    "professional": "professional",
    "friendly": "friendly",
    "casual": "casual",
  }
  return {"formality": mapping.get(tone, "professional"), "tone": tone}


def optimize_tone(
  tone: str,
  intent: str,
  sentiment: dict[str, Any],
  *,
  mode: str = "",
  email_type: str = "",
) -> dict[str, Any]:
  effective = tone
  if mode == "cold_email" and tone == "formal":
    return {"requested_tone": tone, "effective_tone": "formal"}
  if sentiment.get("sentiment") in ("concerned", "neutral_negative") and tone == "casual":
    effective = "professional"
  if intent in ("apologize", "support") and tone == "casual":
    effective = "professional"
  if email_type == "complaint" and tone not in ("formal", "professional"):
    effective = "professional"
  if intent == "sell" and tone == "formal" and mode != "cold_email":
    effective = "professional"
  return {"requested_tone": tone, "effective_tone": effective}


def adapt_culture(locale: str, tone: str) -> dict[str, Any]:
  styles = {
    "en": {"style": "direct_concise", "greeting_weight": "medium", "sign_off": "regards"},
    "hi": {"style": "respectful_warm", "greeting_weight": "high", "sign_off": "thanks"},
    "de": {"style": "direct_structured", "greeting_weight": "medium", "sign_off": "regards"},
    "ja": {"style": "very_respectful", "greeting_weight": "high", "sign_off": "respectfully"},
    "ar": {"style": "formal_respectful", "greeting_weight": "high", "sign_off": "regards"},
  }
  return {"locale": locale, **styles.get(locale, styles["en"])}


def classify_business_domain(text: str) -> dict[str, Any]:
  hay = text.lower()
  scored: list[tuple[int, str]] = []
  for domain, hints in _DOMAIN_HINTS.items():
    hits = sum(1 for h in hints if h in hay)
    if hits:
      scored.append((hits, domain))
  if not scored:
    return {"domains": ["General Business"], "primary_domain": "General Business"}
  scored.sort(reverse=True)
  return {
    "domains": [d for _, d in scored[:4]],
    "primary_domain": scored[0][1],
  }


def plan_structure(mode: str, intent: str, tone: str) -> list[str]:
  if mode == "cold_email":
    return ["greeting", "personalized_opener", "value_proposition", "benefit", "cta", "closing", "signature"]
  if mode == "reply":
    base = ["greeting", "acknowledgment", "response_points", "cta", "closing", "signature"]
    if intent in ("apologize", "support"):
      base.insert(2, "empathy")
    return base
  return ["greeting", "opening", "body", "cta", "closing", "signature"]


def _subject_spam_penalty(subject: str) -> int:
  low = subject.lower()
  penalty = sum(25 for w in _SPAM_WORDS if w in low)
  if subject.isupper() and len(subject) > 10:
    penalty += 20
  if "!!!" in subject or "??" in subject:
    penalty += 15
  return penalty


def _trim_subject(subject: str) -> str:
  s = _clean(subject)
  if len(s) <= SUBJECT_MAX:
    return s
  return s[: SUBJECT_MAX - 3].rsplit(" ", 1)[0] + "..."


def generate_subject_options(ctx: dict[str, Any], mode: str, *, intent: str = "inform", seed: int = 0) -> list[str]:
  candidates: list[str] = []
  if mode == "new_email":
    base = ctx.get("subject") or "Quick update"
    etype = intent
    candidates = [
      base,
      f"Update: {base}"[:SUBJECT_MAX],
      f"{base} — next steps"[:SUBJECT_MAX],
      f"Regarding {base}"[:SUBJECT_MAX],
      f"Action needed: {base}"[:SUBJECT_MAX] if etype == "request" else f"Follow-up: {base}"[:SUBJECT_MAX],
    ]
  elif mode == "reply":
    thread = ctx.get("thread") or {}
    re_subj = thread.get("subject") or "your message"
    candidates = [
      f"Re: {re_subj}"[:SUBJECT_MAX],
      f"Re: {re_subj[:35]} — follow-up"[:SUBJECT_MAX],
      "Re: Thanks for your email",
    ]
  else:
    company = ctx.get("company_name", "your team")
    purpose = (ctx.get("purpose_offer") or "partnership")[:35]
    domain = company.split()[0] if company else "team"
    candidates = [
      f"Quick idea for {company}"[:SUBJECT_MAX],
      f"{purpose} — {domain}"[:SUBJECT_MAX],
      f"Introduction: {company}"[:SUBJECT_MAX],
      f"Potential fit for {company}"[:SUBJECT_MAX],
      f"Question for the {company} team"[:SUBJECT_MAX],
    ]
  scored: list[tuple[int, str]] = []
  for s in candidates:
    s = _trim_subject(s)
    if not s:
      continue
    score = 100 - _subject_spam_penalty(s) - max(0, len(s) - SUBJECT_MAX) * 2
    if len(s) < SUBJECT_MIN:
      score -= 5
    scored.append((score, s))
  scored.sort(reverse=True)
  out: list[str] = []
  seen: set[str] = set()
  for _, s in scored:
    if s.lower() not in seen:
      seen.add(s.lower())
      out.append(s)
  return out[:8] or ["Quick update"]


def select_best_subject(options: list[str], body: str = "") -> str:
  if not options:
    return "Quick update"
  best = options[0]
  best_score = -1
  for s in options:
    score = 100 - _subject_spam_penalty(s)
    if body and s.lower().replace("re: ", "") in body.lower():
      score += 5
    if score > best_score:
      best_score = score
      best = s
  return best


def _pick(pool: list[str], seed: int) -> str:
  return pool[seed % len(pool)] if pool else ""


def _cold_domain_template(domain_name: str) -> dict[str, str]:
  return _COLD_DOMAIN_TEMPLATES.get(domain_name, _COLD_DOMAIN_TEMPLATES["General Business"])


def _localized_phrases(locale: str, tone: str) -> dict[str, list[str]] | None:
  """Return greeting/closing pools for non-English locales, or None for English."""
  if not locale or locale == "en":
    return None
  tone_phrases = _LOCALE_PHRASES.get(locale, {}).get(tone) or _LOCALE_PHRASES.get(locale, {}).get("professional")
  return tone_phrases


def _greeting_closing(tone: str, seed: int, locale: str = "en") -> tuple[str, str]:
  localized = _localized_phrases(locale, tone)
  if localized:
    return _pick(localized["greetings"], seed), _pick(localized["closings"], seed)
  return _pick(_GREETINGS.get(tone, _GREETINGS["professional"]), seed), _pick(
    _CLOSINGS.get(tone, _CLOSINGS["professional"]), seed,
  )


def _opening_new_email(tone: str, subject: str, urgency: str) -> str:
  if urgency == "high":
    return f"I'm reaching out regarding {subject.lower()} — hoping to align quickly."
  if tone == "casual":
    return "Hope you're doing well — quick update below."
  if tone == "formal":
    return f"I am writing in reference to {subject.lower()}."
  if tone == "friendly":
    return "Hope your week is going well! I wanted to share a quick update."
  return "I hope you're doing well. I'm writing with a brief update on the subject above."


def _opening_reply(tone: str, thread: dict[str, Any], sentiment: dict[str, Any]) -> str:
  subj = thread.get("subject", "")
  if sentiment.get("sentiment") in ("concerned", "neutral_negative"):
    return "Thank you for your email — I understand the concern and appreciate you flagging this."
  if subj:
    if tone == "friendly":
      return f"Thanks for your note about {subj.lower()}."
    return f"Thank you for your email regarding {subj.lower()}."
  if tone == "friendly":
    return "Thanks for getting back to me — I appreciate the note."
  return "Thank you for your email."


def _opening_cold(
  tone: str,
  company: str,
  purpose: str,
  domain_name: str,
  *,
  seed: int = 0,
) -> tuple[str, str, str, str]:
  """Return greeting, opener, value intro line, and domain-specific CTA."""
  tmpl = _cold_domain_template(domain_name)
  purpose_clean = purpose.rstrip(".")
  context_line = tmpl["context_line"].format(purpose=purpose_clean)
  if tone == "formal":
    greeting = f"Dear {company} Team,"
    opener = f"I am writing to introduce an opportunity related to {purpose_clean}. {context_line}"
  elif tone == "friendly":
    greeting = "Hi there,"
    opener = (
      f"I came across {company} and thought this might be relevant — {context_line}"
    )
  elif tone == "casual":
    greeting = "Hi,"
    opener = f"I've been following {company} — {context_line}"
  else:
    greeting = "Hello,"
    opener = f"I hope this message finds you well. {context_line}"
  return greeting, opener, tmpl["value_intro"], tmpl["cta"]


def compose_email(
  *,
  mode: str,
  tone: str,
  context: dict[str, Any],
  intent: str,
  seed: int = 0,
  sentiment: dict[str, Any] | None = None,
  urgency: dict[str, Any] | None = None,
  domain: dict[str, Any] | None = None,
  culture: dict[str, Any] | None = None,
) -> str:
  sentiment = sentiment or {}
  urgency_level = (urgency or {}).get("level", "low")
  primary_domain = (domain or {}).get("primary_domain", "General Business")
  locale = (culture or {}).get("locale", "en")
  greeting, closing = _greeting_closing(tone, seed, locale)
  cta = _pick(_CTA_BY_INTENT.get(intent, _CTA_BY_INTENT["inform"]), seed + 3)
  bullets: list[str] = []

  if mode == "new_email":
    if locale == "en":
      greeting = _pick(_GREETINGS.get(tone, _GREETINGS["professional"]), seed)
    subject = context.get("subject", "this topic")
    opener = _opening_new_email(tone, subject, urgency_level)
    bullets = list(context.get("key_points") or [])
    parts = [greeting, "", opener, ""]
  elif mode == "reply":
    if locale == "en":
      greeting = _pick(_GREETINGS.get(tone, _GREETINGS["professional"]), seed)
    thread = context.get("thread") or {}
    opener = _opening_reply(tone, thread, sentiment)
    bullets = list(context.get("reply_points") or [])
    if not bullets:
      bullets = ["I've reviewed your message and outlined my response below."]
    parts = [greeting, "", opener, ""]
  else:
    company = context.get("company_name", "your company")
    purpose = context.get("purpose_offer", "our solution")
    value = context.get("value_proposition", "")
    greeting, opener, value_intro, domain_cta = _opening_cold(
      tone, company, purpose, primary_domain, seed=seed,
    )
    if locale != "en" and tone != "formal":
      loc_greeting, loc_closing = _greeting_closing(tone, seed, locale)
      greeting = loc_greeting
      closing = loc_closing
    cta = domain_cta
    bullets = [value] if value else []
    parts = [greeting, "", opener, ""]
    if bullets and value_intro:
      parts.extend(["", value_intro, ""])

  if bullets:
    if len(bullets) == 1 and mode != "cold_email":
      parts.append(bullets[0])
    else:
      for b in bullets[:6]:
        parts.append(f"• {b}" if not b.startswith("•") else b)
    parts.append("")

  if urgency_level == "high" and cta:
    cta = cta.replace("when you have a moment", "as soon as possible")

  if cta:
    parts.extend([cta, ""])
  parts.extend([closing, "", "[Your Name]"])
  return "\n".join(parts).strip()


def apply_grammar_fixes(text: str) -> tuple[str, list[str]]:
  """Auto-fix common issues; return fixed text and applied fix labels."""
  fixes: list[str] = []
  out = text
  fixed_i = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", out)
  if fixed_i != out:
    out = fixed_i
    fixes.append("capitalized_i")
  collapsed = re.sub(r"[ \t]{2,}", " ", out)
  collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
  if collapsed != out:
    out = collapsed
    fixes.append("normalized_whitespace")
  return out.strip(), fixes


def grammar_check(text: str) -> dict[str, Any]:
  issues: list[str] = []
  if re.search(r"(?<![A-Za-z])i(?![A-Za-z])", text):
    issues.append("lowercase_i")
  if re.search(r"\s{2,}", text):
    issues.append("extra_spaces")
  if re.search(r"[.!?]{2,}", text):
    issues.append("punctuation_repeat")
  if re.search(r"\b(hey dude|lol|wtf|omg)\b", text, re.I):
    issues.append("informal_slang")
  score = max(0, 100 - len(issues) * 12)
  return {"score": score, "issues": issues}


def style_optimize(text: str, tone: str) -> dict[str, Any]:
  """Production polish pass after generation."""
  optimized, fixes = apply_grammar_fixes(text)
  lines = optimized.splitlines()
  cleaned: list[str] = []
  for line in lines:
    if re.match(r"^subject\s*:", line, re.I):
      fixes.append("stripped_inline_subject")
      continue
    cleaned.append(line.rstrip())
  optimized = "\n".join(cleaned).strip()
  if tone == "formal":
    optimized = optimized.replace("Thanks!", "Thank you.")
    optimized = optimized.replace("Hi!", "Hello.")
  return {"text": optimized, "fixes_applied": fixes}


def readability_analyze(text: str) -> dict[str, Any]:
  words = _words(text)
  sents = max(1, len(re.findall(r"[.!?]+", text)))
  avg_sent = words / sents
  passive = len(re.findall(r"\b(is|are|was|were|been|being)\s+\w+ed\b", text, re.I))
  ease = max(0, min(100, int(100 - avg_sent * 2.2 - passive * 3)))
  return {
    "reading_ease": ease,
    "grade_level": "general" if ease > 60 else "advanced",
    "word_count": words,
    "sentence_count": sents,
    "avg_sentence_length": round(avg_sent, 1),
    "passive_voice_hits": passive,
    "reading_time_minutes": max(1, round(words / 200)),
  }


def spam_score(text: str, subject: str = "") -> dict[str, Any]:
  combined = f"{subject} {text}".lower()
  hits = [w for w in _SPAM_WORDS if w in combined]
  caps_ratio = sum(1 for c in subject if c.isupper()) / max(len(subject), 1)
  exclamations = combined.count("!")
  score = min(100, len(hits) * 25 + int(caps_ratio > 0.6) * 20 + min(exclamations, 5) * 4)
  return {
    "spam_score": score,
    "spam_risk": "low" if score < 25 else "medium" if score < 50 else "high",
    "triggers": hits,
  }


def filter_pii(text: str) -> dict[str, Any]:
  redacted = text
  found: list[str] = []
  for pattern, label in _PII_PATTERNS:
    if re.search(pattern, text, re.I):
      found.append(label)
      redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted, flags=re.I)
  return {"redacted_text": redacted, "pii_found": found, "had_pii": bool(found)}


def professionalism_validate(text: str, tone: str) -> dict[str, Any]:
  issues: list[str] = []
  if re.search(r"\b(hey dude|lol|wtf|omg)\b", text, re.I):
    issues.append("informal_language")
  if tone == "formal" and re.search(r"\b(yeah|nope|gonna|kinda)\b", text, re.I):
    issues.append("tone_mismatch")
  if len(text) < 50:
    issues.append("too_short")
  if len(text) > 3500:
    issues.append("too_long")
  score = max(0, 100 - len(issues) * 15)
  return {"score": score, "issues": issues, "passed": score >= 70}


def quality_score(
  grammar: dict[str, Any],
  readability: dict[str, Any],
  spam: dict[str, Any],
  professionalism: dict[str, Any],
) -> dict[str, Any]:
  overall = int(
    grammar["score"] * 0.25
    + readability["reading_ease"] * 0.2
    + (100 - spam["spam_score"]) * 0.15
    + professionalism["score"] * 0.25
    + min(95, 70 + readability["word_count"] // 5) * 0.15
  )
  return {
    "overall": min(100, overall),
    "grammar": grammar["score"],
    "readability": readability["reading_ease"],
    "spam": 100 - spam["spam_score"],
    "professionalism": professionalism["score"],
    "clarity": readability["reading_ease"],
    "engagement": min(95, 65 + readability["word_count"] // 4),
    "completeness": min(100, 60 + readability["word_count"] // 3),
  }


def generate_alternatives(
  subject: str,
  body: str,
  tone: str,
  *,
  mode: str = "new_email",
  context: dict[str, Any] | None = None,
  intent: str = "inform",
) -> list[dict[str, str]]:
  ctx = context or {}
  alts: list[dict[str, str]] = []
  tone_map = {
    "short": "casual" if tone != "formal" else "professional",
    "formal": "formal",
    "friendly": "friendly",
  }
  for label, alt_tone in tone_map.items():
    if label == "short":
      lines = [ln for ln in body.splitlines() if ln.strip()]
      email = "\n".join(lines[:5] + lines[-2:]) if len(lines) > 6 else body
    else:
      email = compose_email(
        mode=mode,
        tone=alt_tone,
        context=ctx,
        intent=intent,
        seed=hash(label) % 97,
      )
    alts.append({
      "variant": label,
      "tone": alt_tone,
      "subject": subject,
      "email": style_optimize(email, alt_tone)["text"],
    })
  return alts


def build_suggestions(
  grammar: dict[str, Any],
  prof: dict[str, Any],
  spam: dict[str, Any],
  readability: dict[str, Any],
) -> list[str]:
  tips: list[str] = []
  if spam.get("spam_risk") != "low":
    tips.append("Reduce promotional language in subject and body to improve deliverability.")
  if readability.get("avg_sentence_length", 0) > 22:
    tips.append("Consider shorter sentences for easier reading on mobile.")
  if prof.get("issues"):
    tips.extend(f"Professionalism: {i}" for i in prof["issues"])
  if grammar.get("issues"):
    tips.extend(f"Grammar: {i}" for i in grammar["issues"])
  if readability.get("word_count", 0) > 250:
    tips.append("Email is long — consider a shorter version for busy recipients.")
  return tips[:6]


def build_llm_refinement_prompts(
  *,
  mode: str,
  tone: str,
  subject: str,
  draft: str,
  context: dict[str, Any],
  intent: dict[str, Any],
  email_type: dict[str, Any],
  recipient: dict[str, Any],
  relationship: dict[str, Any],
  domain: dict[str, Any],
  sentiment: dict[str, Any],
  urgency: dict[str, Any],
  culture: dict[str, Any],
  language: dict[str, Any],
  structure: list[str],
) -> tuple[str, str]:
  """Build system + user prompts using full pipeline metadata."""
  bcp47 = language.get("bcp47", "en")
  lang_name = language.get("language", "English")
  primary_intent = intent.get("primary_intent", "inform")
  urgency_level = urgency.get("level", "low")
  primary_domain = domain.get("primary_domain", "General Business")
  recipient_role = recipient.get("primary_role", "Unknown")
  rel = relationship.get("relationship", "professional_contact")
  warmth = relationship.get("warmth", "neutral")
  email_subtype = email_type.get("subtype") or email_type.get("primary_type", "business")
  culture_style = culture.get("style", "direct_concise")
  key_points = context.get("key_points") or context.get("reply_points") or []
  if isinstance(key_points, list):
    points_text = "\n".join(f"- {p}" for p in key_points[:8])
  else:
    points_text = str(key_points)

  lang_instruction = ""
  if bcp47 != "en":
    lang_instruction = (
      f" Write the entire email body in {lang_name} ({bcp47}). "
      "Keep [Your Name] placeholder in Latin script."
    )

  mode_rules = {
    "new_email": "Preserve all key points. Lead with the most important update.",
    "reply": "Acknowledge the original thread. Address each reply point directly.",
    "cold_email": (
      "Keep it concise (under 150 words). Personalize for the company. "
      "No hype or spam triggers. One clear CTA only."
    ),
  }

  system = (
    "You are an expert B2B email copywriter refining a draft for production use.\n"
    f"Mode: {mode}. Tone: {tone}. Intent: {primary_intent}. "
    f"Urgency: {urgency_level}. Recipient: {recipient_role}. "
    f"Relationship: {rel} ({warmth}). Industry: {primary_domain}. "
    f"Email type: {email_subtype}. Culture style: {culture_style}.\n"
    f"Rules: {mode_rules.get(mode, mode_rules['new_email'])}\n"
    "Keep: greeting, body structure, bullet key points, single CTA, sign-off, [Your Name]. "
    "Do not add subject line, markdown, or fabricated facts/metrics."
    f"{lang_instruction}"
  )

  user_parts = [
    f"Subject: {subject}",
    f"Structure sections: {', '.join(structure)}",
    f"Sentiment context: {sentiment.get('sentiment', 'neutral')}",
  ]
  if mode == "cold_email":
    user_parts.extend([
      f"Company: {context.get('company_name', '')}",
      f"Purpose: {context.get('purpose_offer', '')}",
      f"Value proposition: {context.get('value_proposition', '')}",
    ])
  elif mode == "reply":
    thread = context.get("thread") or {}
    if thread.get("subject"):
      user_parts.append(f"Original subject: {thread['subject']}")
    if context.get("original_email"):
      user_parts.append(f"Original excerpt: {context['original_email'][:400]}")
  else:
    user_parts.append(f"Context summary: {context.get('summary', '')}")

  if points_text:
    user_parts.append(f"Key points to preserve:\n{points_text}")
  user_parts.append(f"\nDraft to refine:\n{draft}")

  return system, "\n".join(user_parts)
