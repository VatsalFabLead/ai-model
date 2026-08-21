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

MULTILINGUAL_EMAIL_CATALOG: dict[str, dict[str, Any]] = {
  "gu": {
    "greetings": {
      "professional": ["નમસ્તે,", "શુભ પ્રભાત / નમસ્કાર,"],
      "formal": ["આદરણીય શ્રી / શ્રીમતી,", "માનનીય ટીમ,"],
      "friendly": ["કેમ છો,", "સ્નેહી મિત્ર,"],
      "casual": ["હેલો,", "કેમ છો,"],
    },
    "closings": {
      "professional": ["આપનો વિશ્વાસુ,", "આભાર સહ,"],
      "formal": ["સાદર પ્રણામ,", "આપનો નમ્ર,"],
      "friendly": ["શુભેચ્છાઓ સાથે,", "આભાર,"],
      "casual": ["આવજો,", "આભાર,"],
    },
    "subject_prefix": "વિષય",
    "soft_cta": "શું આપણે આવતા અઠવાડિયે ૫ મિનિટની ટૂંકી વાતચીત કરી શકીએ?",
    "direct_cta": "શું તમે આ ગુરુવારે બપોરે ૨ વાગ્યે ૧૦ મિનિટના કૉલ માટે ઉપલબ્ધ છો?",
    "value_cta": "શું હું આપને અમારી વિગતવાર સર્વિસ પ્રોફાઇલ/ડેક મોકલી શકું?",
  },
  "hi": {
    "greetings": {
      "professional": ["नमस्ते,", "नमस्कार,"],
      "formal": ["आदरणीय महोदय / महोदया,", "प्रिय टीम,"],
      "friendly": ["हेलो,", "नमस्ते,"],
      "casual": ["हेलो,", "हाय,"],
    },
    "closings": {
      "professional": ["सादर,", "सधन्यवाद,"],
      "formal": ["भवदीय,", "सादर प्रणाम,"],
      "friendly": ["शुभकामनाएं,", "धन्यवाद,"],
      "casual": ["फिर मिलते हैं,", "धन्यवाद,"],
    },
    "subject_prefix": "विषय",
    "soft_cta": "क्या आप अगले सप्ताह ५ मिनट की संक्षिप्त बातचीत के लिए उपलब्ध हैं?",
    "direct_cta": "क्या आप इस गुरुवार दोपहर २ बजे १० मिनट की कॉल के लिए उपलब्ध हैं?",
    "value_cta": "क्या मैं आपको हमारी विस्तृत सर्विस प्रोफाइल/केस स्टडी भेज सकता हूँ?",
  },
  "mr": {
    "greetings": {
      "professional": ["नमस्कार,", "सप्रेम नमस्कार,"],
      "formal": ["आदरणीय महोदय,", "माननीय टीम,"],
      "friendly": ["नमस्कार,", "हॅलो,"],
      "casual": ["हॅलो,"],
    },
    "closings": {
      "professional": ["आपला नम्र,", "सस्नेह धन्यवाद,"],
      "formal": ["सादर प्रणाम,", "आपला स्नेही,"],
      "friendly": ["शुभकामना,", "धन्यवाद,"],
      "casual": ["धन्यवाद,"],
    },
    "subject_prefix": "विषय",
    "soft_cta": "पुढील आठवड्यात ५ मिनिटांच्या संक्षिप्त चर्चेसाठी वेळ मिळेल का?",
    "direct_cta": "या गुरुवारी दुपारी २ वाजता १० मिनिटांच्या कॉलसाठी उपलब्ध आहात का?",
    "value_cta": "मी आमची सविस्तर माहिती पत्रक/केस स्टडी पाठवू का?",
  },
  "es": {
    "greetings": {
      "professional": ["Estimado/a,", "Hola, buen día,"],
      "formal": ["Estimados señores,", "Estimado/a cliente,"],
      "friendly": ["Hola,", "Espero que estés muy bien,"],
      "casual": ["Hola,", "Qué tal,"],
    },
    "closings": {
      "professional": ["Atentamente,", "Un cordial saludo,"],
      "formal": ["Le saluda atentamente,", "Quedo a su disposición,"],
      "friendly": ["Saludos cordiales,", "Un saludo,"],
      "casual": ["Hasta pronto,", "Un abrazo,"],
    },
    "subject_prefix": "Asunto",
    "soft_cta": "¿Estarías libre para una breve charla de 5 minutos la próxima semana?",
    "direct_cta": "¿Tienes disponibilidad para una llamada de 10 minutos este jueves a las 14:00?",
    "value_cta": "¿Te gustaría que te envíe nuestra presentación detallada de casos de éxito?",
  },
  "fr": {
    "greetings": {
      "professional": ["Bonjour,", "Chère équipe,"],
      "formal": ["Madame, Monsieur,", "Cher/Chère client(e),"],
      "friendly": ["Bonjour,", "J'espère que vous allez bien,"],
      "casual": ["Salut,", "Bonjour,"],
    },
    "closings": {
      "professional": ["Cordialement,", "Bien cordialement,"],
      "formal": ["Veuillez agréer mes salutations distinguées,", "Respectueusement,"],
      "friendly": ["Bien à vous,", "Amicalement,"],
      "casual": ["À bientôt,", "Merci,"],
    },
    "subject_prefix": "Objet",
    "soft_cta": "Seriez-vous disponible pour un court échange de 5 minutes la semaine prochaine ?",
    "direct_cta": "Seriez-vous libre pour un appel de 10 minutes ce jeudi à 14h ?",
    "value_cta": "Souhaitez-vous que je vous envoie notre présentation détaillée de cas clients ?",
  },
  "de": {
    "greetings": {
      "professional": ["Guten Tag,", "Hallo,"],
      "formal": ["Sehr geehrte Damen und Herren,", "Sehr geehrte(r) Herr/Frau,"],
      "friendly": ["Hallo,", "Hoffe es geht Dir gut,"],
      "casual": ["Hi,", "Hallo,"],
    },
    "closings": {
      "professional": ["Mit freundlichen Grüßen,", "Beste Grüße,"],
      "formal": ["Mit vorzüglicher Hochachtung,", "Hochachtungsvoll,"],
      "friendly": ["Herzliche Grüße,", "Viele Grüße,"],
      "casual": ["Bis bald,", "Danke,"],
    },
    "subject_prefix": "Betreff",
    "soft_cta": "Hätten Sie nächste Woche Zeit für ein kurzes 5-minütiges Gespräch?",
    "direct_cta": "Passt Ihnen ein 10-minütiger Anruf diesen Donnerstag um 14:00 Uhr?",
    "value_cta": "Soll ich Ihnen unsere ausführliche Fallstudien-Präsentation zusenden?",
  },
  "en": {
    "greetings": {
      "professional": ["Hello,", "Good day,", "Hi there,"],
      "formal": ["Dear Team,", "Dear Sir or Madam,", "To Whom It May Concern,"],
      "friendly": ["Hi,", "Hello there,", "Hope you're doing well —"],
      "casual": ["Hey,", "Hi!", "Hello —"],
    },
    "closings": {
      "professional": ["Best regards,", "Kind regards,", "Thank you,"],
      "formal": ["Sincerely,", "Yours faithfully,", "Respectfully,"],
      "friendly": ["Warm regards,", "Thanks so much,", "Best,"],
      "casual": ["Cheers,", "Thanks,", "Talk soon,"],
    },
    "subject_prefix": "Subject",
    "soft_cta": "Would you be open to a quick 5-minute chat next week?",
    "direct_cta": "Are you free for a 10-minute call this Thursday at 2 PM IST?",
    "value_cta": "Should I send over our detailed project overview and case study deck?",
  },
}

_SPAM_TRIGGER_CATALOG = [
  ("100% free", "complimentary / no cost"),
  ("risk free", "guaranteed quality"),
  ("buy now", "explore details"),
  ("click here", "visit link"),
  ("act now", "time sensitive"),
  ("earn money fast", "increase revenue"),
  ("guaranteed success", "proven track record"),
  ("no obligation", "hassle free"),
  ("winner", "selected partner"),
  ("congratulations", "pleased to connect"),
  ("double your income", "accelerate business growth"),
  ("special promotion", "exclusive offer"),
]

_GREETINGS: dict[str, list[str]] = MULTILINGUAL_EMAIL_CATALOG["en"]["greetings"]
_CLOSINGS: dict[str, list[str]] = MULTILINGUAL_EMAIL_CATALOG["en"]["closings"]


def audit_spam_trigger_words(subject: str, email_body: str) -> dict[str, Any]:
  """Audit subject and body for spam trigger words and provide deliverability score."""
  text = f"{subject} {email_body}".lower()
  detected = []
  total_penalty = 0

  for trigger, replacement in _SPAM_TRIGGER_CATALOG:
    if trigger in text:
      total_penalty += 15
      detected.append({
        "trigger_word": trigger,
        "risk_level": "High" if total_penalty >= 30 else "Medium",
        "suggested_replacement": replacement,
      })

  spam_score = min(100, total_penalty)
  if spam_score == 0:
    rating = "Excellent (Inbox Ready)"
  elif spam_score <= 25:
    rating = "Good (Low Risk)"
  elif spam_score <= 50:
    rating = "Needs Review (Moderate Risk)"
  else:
    rating = "High Spam Filter Risk"

  return {
    "spam_score": spam_score,
    "deliverability_rating": rating,
    "clean_deliverability": spam_score < 30,
    "detected_triggers": detected,
    "trigger_count": len(detected),
  }


def generate_html_email_template(
  subject: str,
  email_body: str,
  cta: str | None = None,
  signature: str | None = None,
) -> str:
  """Generate copy-pasteable responsive HTML email with inline CSS formatting."""
  body_paragraphs = "\n".join(
    f'<p style="margin: 0 0 14px 0;">{p.strip()}</p>'
    for p in email_body.split("\n")
    if p.strip()
  )

  cta_html = ""
  if cta and cta.strip():
    cta_html = (
      '<div style="text-align: center; margin: 25px 0;">\n'
      '  <a href="#" style="background-color: #2563eb; color: #ffffff; padding: 12px 24px; '
      'text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 14px; '
      'display: inline-block;">\n'
      f'    {cta.strip()}\n'
      '  </a>\n'
      '</div>\n'
    )

  sig_html = ""
  if signature and signature.strip():
    sig_paragraphs = "<br/>".join(signature.strip().split("\n"))
    sig_html = (
      '<div style="border-top: 1px solid #e5e7eb; margin-top: 24px; padding-top: 16px; '
      'font-size: 13px; color: #4b5563; line-height: 1.5;">\n'
      f'  {sig_paragraphs}\n'
      '</div>\n'
    )

  return (
    '<!DOCTYPE html>\n'
    '<html>\n'
    '<head>\n'
    '  <meta charset="utf-8"/>\n'
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
    f'  <title>{subject}</title>\n'
    '</head>\n'
    '<body style="margin: 0; padding: 20px; background-color: #f3f4f6; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif;">\n'
    '  <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">\n'
    '    <tr>\n'
    '      <td style="padding: 32px; font-size: 15px; line-height: 1.6; color: #1f2937;">\n'
    f'        {body_paragraphs}\n'
    f'        {cta_html}\n'
    f'        {sig_html}\n'
    '      </td>\n'
    '    </tr>\n'
    '  </table>\n'
    '</body>\n'
    '</html>'
  )


def generate_cta_strategies(
  intent: str,
  domain: str,
  tone: str,
  language: str | None = None,
) -> dict[str, str]:
  """Generate 3 distinct CTA strategy options (Soft, Direct, Value)."""
  lang = (language or "en").lower()[:2]
  catalog = MULTILINGUAL_EMAIL_CATALOG.get(lang, MULTILINGUAL_EMAIL_CATALOG["en"])

  return {
    "soft_cta": catalog["soft_cta"],
    "direct_cta": catalog["direct_cta"],
    "value_cta": catalog["value_cta"],
  }


def generate_ab_subject_buckets(
  subject: str,
  context_text: str,
  language: str | None = None,
) -> dict[str, str]:
  """Group subject line options into 3 psychological A/B testing strategy buckets."""
  sub = subject.strip()
  prefix = (sub.split(":")[0] if ":" in sub else sub).strip()

  return {
    "curiosity_hook": f"Quick question regarding {prefix.lower()}...",
    "benefit_value": f"Accelerate your workflow with {prefix}",
    "direct_action": f"Action Required: {prefix}",
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
    "context_line": "We partner with engineering and product teams to {purpose}.",
    "value_intro": "Key outcomes for technology leaders:",
    "cta": "Would a 10-minute introductory conversation make sense this week?",
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
    "context_line": "We assist growth-focused companies with {purpose}.",
    "value_intro": "Key business benefits we bring to your organization:",
    "cta": "Would you be open to a brief 10-minute conversation to explore this?",
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
  "gu": {
    "professional": {"greetings": ["નમસ્તે,", "શુભ પ્રભાત,"], "closings": ["આપનો વિશ્વાસુ,", "આભાર સહ,"]},
    "formal": {"greetings": ["આદરણીય શ્રી / શ્રીમતી,", "માનનીય ટીમ,"], "closings": ["સાદર પ્રણામ,", "આપનો નમ્ર,"]},
    "friendly": {"greetings": ["કેમ છો,", "સ્નેહી મિત્ર,"], "closings": ["શુભેચ્છાઓ સાથે,", "આભાર,"]},
    "casual": {"greetings": ["હેલો,", "કેમ છો,"], "closings": ["આવજો,", "આભાર,"]},
  },
  "mr": {
    "professional": {"greetings": ["नमस्कार,", "सप्रेम नमस्कार,"], "closings": ["आपला नम्र,", "सस्नेह धन्यवाद,"]},
    "formal": {"greetings": ["आदरणीय महोदय,", "माननीय टीम,"], "closings": ["सादर प्रणाम,", "आपला स्नेही,"]},
    "friendly": {"greetings": ["नमस्कार,", "हॅलो,"], "closings": ["शुभकामना,", "धन्यवाद,"]},
    "casual": {"greetings": ["हॅलो,"], "closings": ["धन्यवाद,"]},
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
  # 1. Unicode script detection
  if re.search(r"[\u0A80-\u0AFF]", text):
    return {"language": "Gujarati", "bcp47": "gu", "source": "unicode_script"}
  if re.search(r"[\u0900-\u097F]", text):
    if any(w in low for w in ("आहे", "आहोत", "करा", "होते", "नाही")):
      return {"language": "Marathi", "bcp47": "mr", "source": "unicode_script"}
    return {"language": "Hindi", "bcp47": "hi", "source": "unicode_script"}
  if re.search(r"[\u0600-\u06FF]", text):
    return {"language": "Arabic", "bcp47": "ar", "source": "unicode_script"}

  hints = {
    "en": ("the", "and", "please", "thank", "regards", "hello", "dear"),
    "es": ("hola", "gracias", "estimado", "saludos", "buenos", "propuesta", "servicio"),
    "fr": ("bonjour", "merci", "cordialement", "madame", "service"),
    "de": ("hallo", "danke", "freundliche", "guten", "angebot"),
    "hi": ("नमस्ते", "धन्यवाद", "आपका", "सादर"),
    "gu": ("નમસ્તે", "આભાર", "ગુજરાતી"),
    "mr": ("नमस्कार", "धन्यवाद"),
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
    "hi": "Hindi", "gu": "Gujarati", "mr": "Marathi", "ar": "Arabic",
    "pt": "Portuguese", "zh": "Chinese",
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


def _clean_cold_phrase(text: str) -> str:
  s = (text or "").strip().lower()
  replacements = {
    "invitation for meeting": "exploring a potential collaboration and introductory discussion",
    "invitation msg": "streamlining outreach communications and boosting engagement rates",
    "meeting": "scheduling a brief introductory discussion",
    "demo": "providing an interactive walkthrough of our solution",
    "sales": "accelerating sales pipeline growth and team efficiency",
    "partnership": "exploring potential partnership opportunities",
    "collaboration": "identifying strategic joint initiatives",
  }
  for k, v in replacements.items():
    if s == k or s == k + "s":
      return v
  return text.strip()


def _expand_cold_value_prop(value: str, purpose: str, company: str) -> list[str]:
  val_raw = (value or "").strip()
  val_clean = _clean_cold_phrase(val_raw)

  if not val_raw or len(val_raw) <= 15 or val_raw.lower() in ("invitation msg", "invitation", "msg", "value prop", "offer", "sales"):
    return [
      f"Streamline team workflows and eliminate communication friction for {company}",
      "Boost outreach engagement and increase response rates across key stakeholders",
      f"Deliver measurable business impact and save valuable team hours at {company}",
    ]

  parts = [p.strip() for p in re.split(r"[\n;,•]+", val_raw) if len(p.strip()) > 2]
  if len(parts) >= 2:
    return [parts[0], parts[1]]

  text_lower = val_clean[0].lower() + val_clean[1:] if val_clean else val_clean
  if text_lower.startswith(("reduce", "increase", "automate", "boost", "drive", "improve", "streamline")):
    return [
      f"{text_lower.capitalize()}",
      f"Drive higher team efficiency and measurable ROI for {company}",
    ]

  return [
    f"Streamline key processes to deliver {text_lower}",
    f"Drive higher team performance and ROI for {company}",
  ]


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
    purpose_raw = ctx.get("purpose_offer") or ""
    candidates = [
      f"Quick question for {company}"[:SUBJECT_MAX],
      f"Exploring a partnership with {company}"[:SUBJECT_MAX],
      f"Ideas for {company}"[:SUBJECT_MAX],
      f"Connecting with {company}"[:SUBJECT_MAX],
      f"Potential fit for {company}"[:SUBJECT_MAX],
      f"Brief intro — {company}"[:SUBJECT_MAX],
      f"Question for the {company} team"[:SUBJECT_MAX],
    ]
    if purpose_raw and len(purpose_raw) > 5 and purpose_raw.lower() not in ("invitation for meeting", "meeting", "invitation", "invitation msg"):
      candidates.insert(0, f"{purpose_raw.capitalize()[:30]} for {company}"[:SUBJECT_MAX])
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


def _opening_new_email(tone: str, subject: str, urgency: str, locale: str = "en") -> str:
  loc = (locale or "en").lower()[:2]
  if loc == "gu":
    return f"હું આપને {subject} સંદર્ભે મહત્વપૂર્ણ માહિતી શેર કરવા માટે લખી રહ્યો છું."
  if loc == "hi":
    return f"मैं आपको {subject} के संबंध में महत्वपूर्ण जानकारी साझा करने के लिए लिख रहा हूँ।"
  if loc == "mr":
    return f"मी आपणास {subject} बाबत महत्त्वाची माहिती देण्याकरिता हे पत्र लिहित आहे."
  if loc == "es":
    return f"Le escribo en relación con {subject.lower()} para compartir información clave."
  if loc == "fr":
    return f"Je vous écris concernant {subject.lower()} afin de vous partager une mise à jour."
  if loc == "de":
    return f"Ich schreibe Ihnen bezüglich {subject.lower()}, um ein wichtiges Update zu teilen."

  if urgency == "high":
    return f"I'm reaching out regarding {subject.lower()} — hoping to align quickly."
  if tone == "casual":
    return "Hope you're doing well — quick update below."
  if tone == "formal":
    return f"I am writing in reference to {subject.lower()}."
  if tone == "friendly":
    return "Hope your week is going well! I wanted to share a quick update."
  return "I hope you're doing well. I'm writing with a brief update on the subject above."


def _opening_reply(tone: str, thread: dict[str, Any], sentiment: dict[str, Any], locale: str = "en") -> str:
  loc = (locale or "en").lower()[:2]
  s_type = sentiment.get("sentiment", "neutral")
  subj = thread.get("subject", "")

  # 1. Multilingual + Sentiment De-escalation & Warmth Matching
  if loc == "gu":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "આપના ઇમેઇલ બદલ આભાર. અમે તમારી ચિંતા સંપૂર્ણપણે સમજીએ છીએ અને આ બાબતને ખૂબ ગંભીરતાથી લઈએ છીએ."
    if s_type in ("positive", "appreciative"):
      return "આપના સુંદર પ્રતિસાદ બદલ ખૂબ ખૂબ આભાર! અમને જાણીને ખૂબ ખુશી થઈ."
    return "આપના ઇમેઇલ બદલ આપનો ખૂબ ખૂબ આભાર."

  if loc == "hi":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "आपके ईमेल के लिए धन्यवाद। हम आपकी चिंता को पूरी तरह समझते हैं और इस मामले को गंभीरता से ले रहे हैं।"
    if s_type in ("positive", "appreciative"):
      return "आपकी सुंदर प्रतिक्रिया के लिए बहुत-बहुत धन्यवाद! हमें यह जानकर बेहद खुशी हुई।"
    return "आपके ईमेल के लिए धन्यवाद।"

  if loc == "mr":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "आपल्या ईमेलबद्दल धन्यवाद. आम्ही आपली अडचण समजतो आणि यावर त्वरित योग्य कारवाई करत आहोत."
    if s_type in ("positive", "appreciative"):
      return "आपल्या उत्तम प्रतिक्रियेबद्दल खूप खूप धन्यवाद! आम्हाला मनापासून आनंद झाला."
    return "आपल्या ईमेलबद्दल धन्यवाद."

  if loc == "es":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "Gracias por su mensaje. Comprendemos perfectamente su preocupación y asumimos este asunto con máxima prioridad."
    if s_type in ("positive", "appreciative"):
      return "¡Muchísimas gracias por sus amables palabras! Nos alegra enormemente saberlo."
    return "Gracias por su correo electrónico."

  if loc == "fr":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "Merci pour votre message. Nous comprenons parfaitement votre inquiétude et nous traitons ce sujet en priorité."
    if s_type in ("positive", "appreciative"):
      return "Merci beaucoup pour vos chaleureux retours ! Nous sommes ravis d'apprendre cela."
    return "Merci pour votre courriel."

  if loc == "de":
    if s_type in ("concerned", "neutral_negative", "frustrated"):
      return "Vielen Dank für Ihre E-Mail. Wir verstehen Ihr Anliegen vollkommen und nehmen diesen Fall sehr ernst."
    if s_type in ("positive", "appreciative"):
      return "Vielen Dank für Ihr positives Feedback! Wir freuen uns sehr darüber."
    return "Vielen Dank für Ihre E-Mail."

  # English Fallback
  if s_type in ("concerned", "neutral_negative", "frustrated"):
    return "Thank you for your email — we sincerely apologize for the inconvenience and take this matter very seriously."
  if s_type in ("positive", "appreciative"):
    return "Thank you so much for your kind words! We are thrilled to hear this and truly appreciate your feedback."
  if subj:
    if tone == "friendly":
      return f"Thanks for your note about {subj.lower()}."
    return f"Thank you for your email regarding {subj.lower()}."
  return "Thank you for your email."


def generate_reply_stances(
  original_email: str,
  reply_points: str,
  tone: str,
  language: str | None = None,
) -> dict[str, str]:
  """Generate 3 pre-built stance alternatives for quick replies (Accept, Decline, Clarify)."""
  lang = (language or "en").lower()[:2]

  if lang == "gu":
    return {
      "accept_stance": "આપનો આભાર! અમે આ દરખાસ્ત સાથે સંતોષ વ્યક્ત કરીએ છીએ અને આગળ વધવા માટે તૈયાર છીએ.",
      "decline_stance": "આપના ઇમેઇલ બદલ આભાર. જો કે, હાલના તબક્કે અમે આ બાબતે આગળ વધી શકીએ તેમ નથી.",
      "clarify_stance": "આપના ઇમેઇલ બદલ આભાર. આગળ વધતા પહેલાં, શું આપ કૃપા કરીને આ બાબતે વધુ સ્પષ્ટતા કરી શકશો?",
    }
  if lang == "hi":
    return {
      "accept_stance": "धन्यवाद! हम इस प्रस्ताव से पूरी तरह सहमत हैं और आगे बढ़ने के लिए तैयार हैं।",
      "decline_stance": "आपके ईमेल के लिए धन्यवाद। हालाँकि, वर्तमान में हम इस प्रस्ताव को स्वीकार करने में असमर्थ हैं।",
      "clarify_stance": "आपके ईमेल के लिए धन्यवाद। आगे बढ़ने से पहले, क्या आप कृपया इस बिंदु पर थोड़ी और स्पष्टता दे सकते हैं?",
    }
  if lang == "es":
    return {
      "accept_stance": "¡Muchas gracias! Estamos de acuerdo con la propuesta y listos para avanzar.",
      "decline_stance": "Gracias por su correo. Lamentablemente, en este momento no nos es posible proceder.",
      "clarify_stance": "Gracias por los detalles. Antes de continuar, ¿podría aclararnos un par de puntos?",
    }

  return {
    "accept_stance": "Thank you! We agree with the proposal and are ready to move forward.",
    "decline_stance": "Thank you for reaching out. However, at this time we are unable to proceed with this request.",
    "clarify_stance": "Thank you for the details. Before moving forward, could you please clarify a few points?",
  }


def generate_html_thread_reply(
  reply_body: str,
  original_email: str,
  sender: str = "Sender",
  subject: str = "Message",
) -> str:
  """Generate standard Gmail/Outlook styled thread reply HTML with blockquote divider."""
  reply_html = "\n".join(
    f'<p style="margin: 0 0 14px 0;">{p.strip()}</p>'
    for p in reply_body.split("\n")
    if p.strip()
  )

  quoted_html = "<br/>".join(
    p.strip() for p in (original_email or "").split("\n") if p.strip()
  )

  return (
    '<!DOCTYPE html>\n'
    '<html>\n'
    '<head>\n'
    '  <meta charset="utf-8"/>\n'
    '  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>\n'
    '</head>\n'
    '<body style="margin: 0; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif; font-size: 15px; color: #1f2937; line-height: 1.6;">\n'
    '  <!-- New Reply Body -->\n'
    f'  <div style="margin-bottom: 24px;">\n'
    f'    {reply_html}\n'
    '  </div>\n'
    '  <!-- Gmail / Outlook Quoted Original Thread -->\n'
    '  <div style="border-left: 2px solid #cbd5e1; margin-top: 24px; padding-left: 14px; color: #64748b; font-size: 13px; line-height: 1.5;">\n'
    f'    <p style="margin: 0 0 8px 0; font-weight: 600; color: #475569;">On Re: {subject}, {sender} wrote:</p>\n'
    '    <blockquote style="margin: 0; padding: 0;">\n'
    f'      {quoted_html}\n'
    '    </blockquote>\n'
    '  </div>\n'
    '</body>\n'
    '</html>'
  )


def audit_thread_reply_coverage(original_email: str, reply_body: str) -> dict[str, Any]:
  """Audit incoming email questions and verify if they were addressed in the reply."""
  sentences = re.split(r"[.!?\n]+", original_email or "")
  questions = [
    s.strip() for s in sentences
    if len(s.strip()) > 8 and ("?" in s or any(w in s.lower() for w in ("how", "what", "when", "cost", "price", "where", "why", "કેવી રીતે", "શું", "कैसे", "क्या")))
  ]

  reply_low = (reply_body or "").lower()
  answered = []
  for q in questions:
    keywords = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 3]
    if any(kw in reply_low for kw in keywords):
      answered.append(q)

  total_q = len(questions)
  answered_q = len(answered)
  coverage = round((answered_q / total_q) * 100) if total_q > 0 else 100

  return {
    "incoming_questions_found": questions,
    "answered_questions": answered,
    "total_questions": total_q,
    "answered_count": answered_q,
    "coverage_score": coverage,
    "all_questions_addressed": coverage >= 80,
  }


def _opening_cold(
  tone: str,
  company: str,
  purpose: str,
  domain_name: str,
  *,
  seed: int = 0,
  locale: str = "en",
) -> tuple[str, str, str, str]:
  """Return localized greeting, opener, value intro line, and domain-specific CTA."""
  loc = (locale or "en").lower()[:2]
  tmpl = _cold_domain_template(domain_name)
  purpose_clean = _clean_cold_phrase(purpose)

  if loc == "gu":
    greeting = f"માનનીય {company} ટીમ,"
    opener = f"હું આપની સંસ્થા {company} સાથે {purpose_clean} સંદર્ભે વ્યાવસાયિક જોડાણ માટે સંપર્ક કરી રહ્યો છું."
    value_intro = f"{company} જેવી અગ્રણી સંસ્થાઓ માટે મુખ્ય ફાયદાઓ:"
    cta = "શું આવતા અઠવાડિયે ૧૦ મિનિટની ટૂંકી વાતચીત શક્ય બનશે?"
    return greeting, opener, value_intro, cta

  if loc == "hi":
    greeting = f"प्रिय {company} टीम,"
    opener = f"मैं आपकी संस्था {company} के साथ {purpose_clean} के संबंध में संपर्क कर रहा हूँ।"
    value_intro = f"{company} जैसी प्रमुख संस्थाओं के लिए मुख्य लाभ:"
    cta = "क्या अगले सप्ताह १० मिनट की संक्षिप्त बातचीत संभव होगी?"
    return greeting, opener, value_intro, cta

  if loc == "mr":
    greeting = f"मा. {company} टीम,"
    opener = f"मी आपल्या {company} सोबत {purpose_clean} या विषयावर व्यवसाय चर्चेसाठी संपर्क साधत आहे."
    value_intro = f"{company} सारख्या संस्थांसाठी मुख्य फायदे:"
    cta = "पुढील आठवड्यात १० मिनिटांच्या चर्चेसाठी वेळ मिळेल का?"
    return greeting, opener, value_intro, cta

  if loc == "es":
    greeting = f"Estimado equipo de {company},"
    opener = f"Le escribo para presentar una oportunidad estratégica para {company} en relación con {purpose_clean}."
    value_intro = f"Beneficios clave para empresas como {company}:"
    cta = "¿Tendría disponibilidad para una breve conversación de 10 minutos la próxima semana?"
    return greeting, opener, value_intro, cta

  if loc == "fr":
    greeting = f"Chère équipe de {company},"
    opener = f"Je vous contacte pour échanger sur une opportunité stratégique pour {company} concernant {purpose_clean}."
    value_intro = f"Bénéfices clés pour des organisations comme {company} :"
    cta = "Seriez-vous disponible pour un court échange de 10 minutes la semaine prochaine ?"
    return greeting, opener, value_intro, cta

  if loc == "de":
    greeting = f"Sehr geehrtes {company}-Team,"
    opener = f"Ich kontaktiere Sie bezüglich einer strategischen Möglichkeit für {company} im Bereich {purpose_clean}."
    value_intro = f"Wichtigste Vorteile für Unternehmen wie {company}:"
    cta = "Hätten Sie nächste Woche Zeit für ein kurzes 10-minütiges Gespräch?"
    return greeting, opener, value_intro, cta

  # English Default
  if tone == "formal":
    greeting = f"Dear {company} Team,"
    opener = f"I am writing to introduce a strategic opportunity regarding {purpose_clean}."
  elif tone == "friendly":
    greeting = f"Hi {company} Team,"
    opener = f"I came across {company} and wanted to reach out regarding {purpose_clean}."
  elif tone == "casual":
    greeting = "Hi there,"
    opener = f"Quick note for the {company} team regarding {purpose_clean}."
  else:  # professional
    greeting = f"Hello {company} Team,"
    opener = f"I am reaching out to {company} to discuss {purpose_clean}."
  value_intro = f"Key business benefits we bring to organizations like {company}:"
  cta = tmpl.get("cta") or f"Would you be open to a brief 10-minute conversation next week to explore if this aligns with {company}'s goals?"
  return greeting, opener, value_intro, cta


def generate_cold_sequence(
  company_name: str,
  purpose_offer: str,
  value_proposition: str,
  tone: str = "professional",
  language: str | None = None,
) -> dict[str, dict[str, str]]:
  """Generate a complete 3-step automated cold outreach campaign sequence."""
  lang = (language or "en").lower()[:2]
  c_name = company_name or "your team"

  if lang == "gu":
    return {
      "step_1": {
        "timing": "Day 1",
        "subject": f"{c_name} માટે મહત્વપૂર્ણ પ્રસ્તાવ",
        "body": f"માનનીય {c_name} ટીમ,\n\nhું આપની સંસ્થા સાથે {purpose_offer} સંદર્ભે વાતચીત કરવા આતુર છું. અમે {value_proposition} માં મદદ કરીએ છીએ.\n\nશું આવતા અઠવાડિયે ૧૦ મિનિટની વાતચીત શક્ય બનશે?\n\nઆપનો વિશ્વાસુ,",
      },
      "step_2": {
        "timing": "Day 3 (Nudge & Proof)",
        "subject": f"Re: {c_name} માટે મહત્વપૂર્ણ પ્રસ્તાવ",
        "body": f"હેલો {c_name} ટીમ,\n\nhું ફક્ત મારો અગાઉનો ઇમેઇલ ફોલોઅપ કરી રહ્યો છું. અમારી સોલ્યુશનથી સમાન કંપનીઓએ કામગીરીમાં ૩૫% નો સુધારો નોંધાવ્યો છે.\n\nશું આપ આ અંગે વધુ જાણવા માંગો છો?\n\nઆભાર,",
      },
      "step_3": {
        "timing": "Day 7 (Breakup Email)",
        "subject": f"અંતિમ સમીક્ષા — {c_name}",
        "body": f"હેલો {c_name} ટીમ,\n\nજો આ સમય આપના માટે અનુકૂળ ન હોય તો હું સમજી શકું છું. ભવિષ્યમાં જરૂર જણાય ત્યારે આપ સંપર્ક કરી શકો છો.\n\nશુભેચ્છાઓ સાથે,",
      },
    }

  return {
    "step_1": {
      "timing": "Day 1",
      "subject": f"Quick question for {c_name}",
      "body": f"Hi {c_name} Team,\n\nI am reaching out to explore how we can assist {c_name} with {purpose_offer}. We specialize in delivering {value_proposition}.\n\nWould you be open to a 10-minute call next week?\n\nBest regards,",
    },
    "step_2": {
      "timing": "Day 3 (Nudge & Social Proof)",
      "subject": f"Re: Quick question for {c_name}",
      "body": f"Hi {c_name} Team,\n\nFollowing up on my previous note. Similar teams in your sector have seen a 35% increase in efficiency using our approach.\n\nShould I send over a quick case study deck?\n\nBest regards,",
    },
    "step_3": {
      "timing": "Day 7 (Breakup Email)",
      "subject": f"Final check-in — {c_name}",
      "body": f"Hi {c_name} Team,\n\nShould I assume this isn't a priority right now? No worries at all if so — I won't crowd your inbox.\n\nFeel free to reach out whenever timing is better.\n\nBest,",
    },
  }


def generate_merge_tag_template(email_body: str, company_name: str) -> str:
  """Generate standardized merge-tag template for cold email tools (Lemlist, Instantly, Salesloft)."""
  text = email_body
  if company_name and company_name in text:
    text = text.replace(company_name, "{{COMPANY_NAME}}")
  text = re.sub(r"^(Hello|Hi|Dear|Hey)\s+[^\n,]+,", r"\1 {{FIRST_NAME}},", text, flags=re.I)
  return text


def generate_cold_strategy_angles(
  company_name: str,
  purpose_offer: str,
  value_proposition: str,
  language: str | None = None,
) -> dict[str, str]:
  """Generate 3 distinct cold outreach strategy angles (Pain Point, Social Proof, Direct ROI)."""
  c_name = company_name or "your team"
  return {
    "pain_point_angle": f"Noticed many teams in {c_name}'s domain struggle with {purpose_offer}. Here is how we help eliminate that bottleneck...",
    "social_proof_angle": f"How we helped a peer organization similar to {c_name} achieve 40% growth in 90 days with {value_proposition}...",
    "direct_roi_angle": f"Reduce operational overhead for {c_name} by 35% while accelerating {purpose_offer}...",
  }


def audit_cold_deliverability(subject: str, email_body: str) -> dict[str, Any]:
  """Audit cold email deliverability metrics (word count, link count, caps, spam triggers)."""
  text = f"{subject} {email_body}"
  words = re.findall(r"\b[\w'-]+\b", text)
  word_count = len(words)

  links = re.findall(r"https?://\S+", text)
  link_count = len(links)

  all_caps_words = [w for w in words if w.isupper() and len(w) > 3 and w not in ("API", "ROI", "B2B", "SEO", "CRM", "CTA", "HTML", "PHP", "PDF", "SaaS")]

  spam_words = [
    "guaranteed roi", "100% free", "risk free", "buy now", "click here",
    "earn money fast", "special offer", "no cost", "instant results",
  ]
  text_low = text.lower()
  found_spam = [w for w in spam_words if w in text_low]

  score = 100
  if word_count > 150:
    score -= 15
  if link_count > 1:
    score -= 20
  if all_caps_words:
    score -= 15
  if found_spam:
    score -= len(found_spam) * 20

  score = max(0, score)
  status = "Optimal Cold Email (High Open & Response Rate)" if score >= 80 else "Needs Trimming / High Spam Risk"

  return {
    "word_count": word_count,
    "is_concise": word_count <= 150,
    "link_count": link_count,
    "has_all_caps": bool(all_caps_words),
    "all_caps_words": all_caps_words,
    "spam_words_found": found_spam,
    "cold_deliverability_score": score,
    "deliverability_status": status,
  }


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
    opener = _opening_new_email(tone, subject, urgency_level, locale)
    bullets = list(context.get("key_points") or [])
    parts = [greeting, "", opener, ""]
  elif mode == "reply":
    if locale == "en":
      greeting = _pick(_GREETINGS.get(tone, _GREETINGS["professional"]), seed)
    thread = context.get("thread") or {}
    opener = _opening_reply(tone, thread, sentiment, locale)
    bullets = list(context.get("reply_points") or [])
    if not bullets:
      bullets = ["I've reviewed your message and outlined my response below."]
    parts = [greeting, "", opener, ""]
  else:
    company = context.get("company_name", "your company")
    purpose = context.get("purpose_offer", "our solution")
    value = context.get("value_proposition", "")
    greeting, opener, value_intro, domain_cta = _opening_cold(
      tone, company, purpose, primary_domain, seed=seed, locale=locale,
    )
    if locale != "en" and tone != "formal":
      loc_greeting, loc_closing = _greeting_closing(tone, seed, locale)
      greeting = loc_greeting
      closing = loc_closing
    cta = domain_cta
    bullets = _expand_cold_value_prop(value, purpose, company)
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
      "CRITICAL COLD EMAIL REQUIREMENTS:\n"
      "1. Mention the target company naturally.\n"
      "2. NEVER repeat raw user input text word-for-word (expand generic terms like 'invitation for meeting' or 'invitation msg' into articulate business benefits and value propositions).\n"
      "3. Expand the value proposition into 2-3 clear business benefits.\n"
      "4. Keep the total email body strictly under 180 words.\n"
      "5. Include a single strong call-to-action.\n"
      "6. Maintain the selected tone.\n"
      "7. NEVER use generic clichés like 'I hope this message finds you well'."
    ),
  }

  system = (
    "You are an expert B2B cold email copywriter refining a draft for production use.\n"
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
      "Requirements:",
      "- Mention company naturally.",
      "- Never repeat raw user text word-for-word.",
      "- Expand value proposition into clear business benefits.",
      "- Keep under 180 words.",
      "- Include single strong CTA.",
      "- Avoid 'I hope this message finds you well'.",
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
