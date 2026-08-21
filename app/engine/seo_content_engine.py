from __future__ import annotations

"""Advanced SEO content engine — worldwide, multilingual, category-aware.

Uses data/seo_content_knowledge.jsonl for training guidance.
100% custom — no GPT, Claude, Gemini.
"""

from app.engine.seo_content_domains import detect_language, get_language_name, get_localized_heading

from pathlib import Path
from typing import Any

import re

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEO_CONTENT_KB_PATH = PROJECT_ROOT / "data" / "seo_content_knowledge.jsonl"

_CATEGORIES: dict[str, dict[str, Any]] = {
  "blog_article": {
    "label": "Blog Article",
    "description": "Long-form SEO blog posts and guides",
    "default_tone": "professional",
  },
  "how_to_guide": {
    "label": "How-To Guide",
    "description": "Step-by-step tutorials that rank for informational queries",
    "default_tone": "professional",
  },
  "listicle": {
    "label": "Listicle",
    "description": "Numbered tips, tools, or best-of articles",
    "default_tone": "casual",
  },
  "landing_page": {
    "label": "Landing Page Copy",
    "description": "Conversion-focused service or product landing pages",
    "default_tone": "professional",
  },
  "product_description": {
    "label": "Product Description",
    "description": "E-commerce and SaaS product page copy",
    "default_tone": "professional",
  },
  "local_seo": {
    "label": "Local SEO",
    "description": "City/region-focused pages for local businesses",
    "default_tone": "friendly",
  },
  "news_update": {
    "label": "News / Update",
    "description": "Timely announcements and industry news posts",
    "default_tone": "formal",
  },
  "ecommerce": {
    "label": "E-commerce SEO",
    "description": "Category pages, buying guides, comparison content",
    "default_tone": "professional",
  },
}

# Only these four tones are supported (matches product UI).
_VALID_TONES = ["professional", "casual", "friendly", "formal"]

_TONE_HINTS: dict[str, str] = {
  "professional": "Clear, confident, and business-appropriate. Polished but accessible.",
  "casual": "Relaxed and conversational. Approachable language, easy to read.",
  "friendly": "Warm, helpful, and welcoming. Supportive voice that builds trust.",
  "formal": "Structured and respectful. Suited for corporate, legal, or academic audiences.",
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en", "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
  "german": "de", "de": "de", "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar", "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh", "korean": "ko", "ko": "ko",
  "italian": "it", "it": "it", "russian": "ru", "ru": "ru",
  "bengali": "bn", "bn": "bn", "tamil": "ta", "ta": "ta",
  "marathi": "mr", "mr": "mr", "urdu": "ur", "ur": "ur",
  "vietnamese": "vi", "vi": "vi", "thai": "th", "th": "th",
  "dutch": "nl", "nl": "nl", "polish": "pl", "pl": "pl",
  "turkish": "tr", "tr": "tr", "indonesian": "id", "id": "id",
}

_seo_kb: KnowledgeBase | None = None


def get_seo_kb() -> KnowledgeBase:
  global _seo_kb
  if _seo_kb is None:
    _seo_kb = load_knowledge_base(knowledge_path=SEO_CONTENT_KB_PATH)
  return _seo_kb


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def normalize_tone(tone: str | None, category: str | None = None) -> str:
  if tone:
    t = tone.strip().lower()
    if t in _VALID_TONES:
      return t
  cat = normalize_category(category)
  default = _CATEGORIES[cat]["default_tone"]
  return default if default in _VALID_TONES else "professional"


def tone_hint(tone: str) -> str:
  return _TONE_HINTS.get(tone, _TONE_HINTS["professional"])


def normalize_category(category: str | None) -> str:
  if not category:
    return "blog_article"
  key = category.strip().lower().replace(" ", "_").replace("-", "_")
  if key in _CATEGORIES:
    return key
  aliases = {
    "blog": "blog_article", "article": "blog_article", "guide": "how_to_guide",
    "howto": "how_to_guide", "landing": "landing_page", "product": "product_description",
    "local": "local_seo", "news": "news_update", "ecommerce": "ecommerce",
  }
  return aliases.get(key, "blog_article")


def supported_categories() -> list[dict[str, str]]:
  return [
    {"id": k, "label": v["label"], "description": v["description"], "default_tone": v["default_tone"]}
    for k, v in _CATEGORIES.items()
  ]


def supported_tones() -> list[dict[str, str]]:
  return [
    {"id": t, "label": t.capitalize()}
    for t in _VALID_TONES
  ]


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"}, {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"}, {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"}, {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"}, {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"}, {"name": "Korean", "code": "ko"},
    {"name": "Italian", "code": "it"}, {"name": "Russian", "code": "ru"},
    {"name": "Bengali", "code": "bn"}, {"name": "Tamil", "code": "ta"},
    {"name": "Marathi", "code": "mr"}, {"name": "Urdu", "code": "ur"},
    {"name": "Vietnamese", "code": "vi"}, {"name": "Thai", "code": "th"},
    {"name": "Dutch", "code": "nl"}, {"name": "Polish", "code": "pl"},
    {"name": "Turkish", "code": "tr"}, {"name": "Indonesian", "code": "id"},
  ]


def get_guidance(topic: str, category: str, language: str | None) -> str:
  kb = get_seo_kb()
  lang = bcp47(language)
  queries = [
    f"SEO content {category} best practices",
    f"SEO article writing {topic}",
    f"SEO content multilingual {lang}",
  ]
  chunks: list[str] = []
  for q in queries:
    answer, score = kb.search(q)
    if answer and score > 0.05 and answer not in chunks:
      chunks.append(answer)
  return "\n\n".join(chunks[:2])


def category_structure_hint(category: str) -> str:
  hints = {
    "blog_article": "Use intro, 3-5 H2 sections, bullet lists where helpful, and a conclusion with CTA.",
    "how_to_guide": "Use numbered steps under H2/H3, prerequisites section, and FAQ-style tips.",
    "listicle": "Use H2 for each list item (e.g. '## 1. First tip'), short paragraphs, summary table optional.",
    "landing_page": "Use hero value prop, benefits bullets, social proof section, and strong CTA.",
    "product_description": "Use features, benefits, use cases, specs, and trust signals.",
    "local_seo": "Mention location naturally, local benefits, service area, and contact CTA.",
    "news_update": "Lead with the key update, context, impact, and what readers should do next.",
    "ecommerce": "Use buying guide structure, comparison points, pros/cons, and recommendation.",
  }
  return hints.get(category, hints["blog_article"])


def build_outline(topic: str, keywords: list[str], category: str, *, language: str | None = None) -> list[str]:
  """Article section outline (H2-level) with full multilingual support."""
  lang = detect_language(topic, keywords, language)
  primary = (keywords[0] if keywords else topic).strip().title()

  if lang == "hi":
    if category == "how_to_guide":
      return [
        f"{primary} शुरू करने से पहले आवश्यक जानकारी",
        f"चरण 1: {primary} के मूल सिद्धांतों को समझें",
        "चरण 2: सर्वोत्तम कार्यप्रणालियों को लागू करें",
        "चरण 3: परिणामों का अनुकूलन और विस्तार करें",
        "सामान्य गलतियाँ और उनसे कैसे बचें",
        "निष्कर्ष और अगले कदम",
      ]
    if category == "listicle":
      return [
        f"1. अपनी {primary} रणनीति को परिभाषित करें",
        f"2. {primary} के लिए सही टूल्स चुनें",
        "3. उच्च गुणवत्ता वाली सामग्री बनाएं",
        "4. निरंतर मापें और सुधार करें",
        "5. नवीनतम रुझानों से अपडेट रहें",
        "सारांश और कार्य योजना",
      ]
    if category == "landing_page":
      return [
        "मुख्य मूल्य प्रस्ताव (Hero Value Proposition)",
        "प्रमुख लाभ और उपयोगिता",
        "यह कैसे काम करता है",
        "विश्वास संकेत और समीक्षाएं",
        "आगे बढ़ें (Call to Action)",
      ]
    return [
      f"{primary}: एक संपूर्ण अवलोकन",
      f"आज {primary} का महत्व क्यों है?",
      f"{primary} के मुख्य लाभ",
      "सर्वोत्तम कार्यप्रणालियाँ और रणनीतियाँ",
      f"{primary} के साथ शुरुआत कैसे करें",
      "निष्कर्ष",
    ]

  if lang == "es":
    if category == "how_to_guide":
      return [
        f"Lo que necesita antes de comenzar con {primary}",
        f"Paso 1: Comprender los conceptos básicos de {primary}",
        "Paso 2: Aplicar mejores prácticas",
        "Paso 3: Optimizar y escalar resultados",
        "Errores comunes a evitar",
        "Conclusión y próximos pasos",
      ]
    return [
      f"Introducción a {primary}",
      f"Por qué es importante {primary} hoy en día",
      f"Beneficios clave de {primary}",
      "Mejores prácticas y estrategias",
      f"Cómo empezar con {primary}",
      "Conclusión",
    ]

  if lang == "fr":
    return [
      f"Introduction à {primary}",
      f"Pourquoi {primary} est important aujourd'hui",
      f"Principaux avantages de {primary}",
      "Meilleures pratiques et stratégies",
      f"Comment démarrer avec {primary}",
      "Conclusion et prochaines étapes",
    ]

  if lang == "de":
    return [
      f"Einführung in {primary}",
      f"Warum {primary} heute wichtig ist",
      f"Hauptvorteile von {primary}",
      "Bewährte Verfahren und Strategien",
      f"Erste Schritte mit {primary}",
      "Fazit und nächste Schritte",
    ]

  if lang == "mr":
    return [
      f"{primary}: सविस्तर परिचय",
      f"आज {primary} चे महत्व का आहे?",
      f"{primary} चे मुख्य फायदे",
      "उत्कृष्ट पद्धती आणि धोरणे",
      f"{primary} सह सुरुवात कशी करावी",
      "निष्कर्ष",
    ]

  if lang == "bn":
    return [
      f"{primary}: প্রাথমিক পরিচিতি",
      f"বর্তমানে {primary}-এর গুরুত্ব কেন?",
      f"{primary}-এর প্রধান সুবিধাসমূহ",
      "সেরা অনুশীলন এবং কৌশলসমূহ",
      f"{primary} দিয়ে কীভাবে শুরু করবেন",
      "উপসংহার",
    ]

  if lang == "gu":
    return [
      f"{primary}: સચોટ પરિચય અને માર્ગદર્શિકા",
      f"શા માટે {primary} મહત્વપૂર્ણ છે?",
      f"{primary} ના મુખ્ય ફાયદાઓ",
      "ઉત્તમોત્તમ પદ્ધતિઓ અને ટિપ્સ",
      f"{primary} સાથે શરૂઆત કેવી રીતે કરવી",
      "નિષ્કર્ષ અને આગળના પગલાં",
    ]

  if lang != "en":
    # Universal fallback for any non-English language (Gujarati, Punjabi, Telugu, Kannada, Malayalam, etc.)
    return [
      f"{primary}: {get_localized_heading('intro', lang)}",
      f"{get_localized_heading('what_is', lang)} {primary}",
      f"{primary} - {get_localized_heading('benefits', lang)}",
      f"{get_localized_heading('how_it_works', lang)}",
      f"{get_localized_heading('guide', lang)}",
      f"{get_localized_heading('conclusion', lang)}",
    ]

  if category == "how_to_guide":
    return [
      f"What You Need Before Starting With {primary}",
      f"Step 1: Understand {primary} Fundamentals",
      "Step 2: Apply Proven Techniques",
      "Step 3: Optimize and Scale Results",
      "Common Mistakes to Avoid",
      "Conclusion and Next Steps",
    ]
  if category == "listicle":
    return [
      f"1. Define Your {primary} Strategy",
      f"2. Choose the Right Tools for {primary}",
      "3. Create High-Quality Original Content",
      "4. Measure and Improve Continuously",
      "5. Stay Updated With Industry Trends",
      "Summary and Action Plan",
    ]
  if category == "landing_page":
    return [
      "Hero Value Proposition",
      "Key Benefits",
      "How It Works",
      "Social Proof and Trust Signals",
      "Call to Action",
    ]
  return [
    f"Introduction to {primary}",
    f"Why {primary} Matters Today",
    f"Key Benefits of {primary}",
    "Best Practices for Worldwide Audiences",
    f"How to Get Started With {primary}",
    "Conclusion",
  ]


def build_faqs(topic: str, keywords: list[str], *, language: str | None = None) -> list[dict[str, str]]:
  """FAQ list — question + answer pairs localized to target language."""
  lang = detect_language(topic, keywords, language)
  primary = (keywords[0] if keywords else topic).strip()

  if lang == "hi":
    return [
      {
        "question": f"{primary} क्या है?",
        "answer": f"{primary} एक महत्वपूर्ण तकनीक और कार्यपद्धति है जिसका उद्देश्य बेहतर प्रदर्शन और सफलता प्राप्त करना है।",
      },
      {
        "question": f"{primary} से परिणाम दिखने में कितना समय लगता है?",
        "answer": "सही रणनीति और निरंतर प्रयास के साथ, 4 से 8 सप्ताह के भीतर सकारात्मक परिणाम दिखाई देने लगते हैं।",
      },
      {
        "question": f"{primary} का उपयोग किसे करना चाहिए?",
        "answer": "यह व्यवसाय स्वामियों, पेशेवरों और उन सभी के लिए उपयोगी है जो अपनी डिजिटल उपस्थिति और दक्षता बढ़ाना चाहते हैं।",
      },
      {
        "question": f"{primary} के मुख्य लाभ क्या हैं?",
        "answer": "मुख्य लाभों में उच्च उत्पादकता, बेहतर गुणवत्ता, लागत प्रभावी निष्पादन और मापने योग्य परिणाम शामिल हैं।",
      },
      {
        "question": f"{primary} की शुरुआत कैसे करें?",
        "answer": "अपनी आवश्यकताओं का विश्लेषण करें, सही टूल्स का चयन करें और एक चरणबद्ध योजना का पालन करें।",
      },
    ]

  if lang == "gu":
    return [
      {
        "question": f"{primary} શું છે?",
        "answer": f"{primary} એ એક મહત્વપૂર્ણ વિષય છે જેના દ્વારા સરળતાથી સફળતા મેળવી શકાય છે.",
      },
      {
        "question": f"{primary} ના મુખ્ય ફાયદાઓ કયા છે?",
        "answer": "આનાથી સમયની બચત થાય છે અને કાર્યમાં ઉત્કૃષ્ટ પરિણામો મળે છે.",
      },
      {
        "question": f"{primary} સાથે શરૂઆત કેવી રીતે કરવી?",
        "answer": "મૂળભૂત બાબતો શીખીને તબક્કાવાર આયોજન સાથે આગળ વધો.",
      },
    ]

  if lang == "es":
    return [
      {
        "question": f"¿Qué es {primary}?",
        "answer": f"{primary} es una estrategia comprobada utilizada para mejorar la visibilidad y obtener resultados medibles.",
      },
      {
        "question": f"¿Cuánto tiempo tarda {primary} en mostrar resultados?",
        "answer": "La mayoría de las estrategias muestran un progreso significativo en un plazo de 4 a 8 semanas con una ejecución constante.",
      },
      {
        "question": f"¿Quién debería enfocarse en {primary}?",
        "answer": "Es ideal para profesionales, emprendedores y empresas que buscan optimizar su rendimiento y crecimiento.",
      },
    ]

  if lang == "fr":
    return [
      {
        "question": f"Qu'est-ce que {primary} ?",
        "answer": f"{primary} est une stratégie éprouvée utilisée par les professionnels pour améliorer la visibilité et les résultats.",
      },
      {
        "question": f"Combien de temps faut-il pour voir les résultats de {primary} ?",
        "answer": "La plupart des stratégies donnent des résultats mesurables en 4 à 8 semaines avec une exécution régulière.",
      },
    ]

  if lang == "de":
    return [
      {
        "question": f"Was ist {primary}?",
        "answer": f"{primary} ist ein bewährter Ansatz zur Verbesserung der Sichtbarkeit und für messbare Ergebnisse.",
      },
      {
        "question": f"Wie lange dauert es, bis {primary} Ergebnisse zeigt?",
        "answer": "Die meisten Strategien zeigen bei konsequenter Umsetzung innerhalb von 4–8 Wochen deutliche Fortschritte.",
      },
    ]

  if lang == "mr":
    return [
      {
        "question": f"{primary} म्हणजे काय?",
        "answer": f"{primary} ही एक प्रभावी पद्धत आहे ज्याचा वापर सर्वोत्तम निकाल मिळवण्यासाठी केला जातो.",
      },
      {
        "question": f"{primary} चे निकाल दिसण्यासाठी किती वेळ लागतो?",
        "answer": "योग्य नियोजन आणि सातत्याने ४ ते ८ आठवड्यात सकारात्मक परिणाम दिसू लागतात.",
      },
    ]

  if lang == "bn":
    return [
      {
        "question": f"{primary} কী?",
        "answer": f"{primary} হলো একটি প্রমাণিত পদ্ধতি যা সাফল্য এবং কার্যকারিতা বৃদ্ধির জন্য ব্যবহৃত হয়।",
      },
      {
        "question": f"{primary}-এর ফলাফল দেখতে কত সময় লাগে?",
        "answer": "সঠিক পরিকল্পনা এবং ধারাবাহিকতার সাথে ৪ থেকে ৮ সপ্তাহের মধ্যে ইতিবাচক ফলাফল পাওয়া যায়।",
      },
    ]

  if lang != "en":
    what_is = get_localized_heading("what_is", lang)
    benefits = get_localized_heading("benefits", lang)
    return [
      {
        "question": f"{what_is} {primary}?",
        "answer": f"{primary} - {topic}",
      },
      {
        "question": f"{primary} - {benefits}?",
        "answer": f"{primary} - {get_localized_heading('key_takeaways', lang)}",
      },
    ]

  return [
    {
      "question": f"What is {primary}?",
      "answer": f"{primary.title()} refers to practical methods and knowledge related to {topic}, applied step by step for real results.",
    },
    {
      "question": f"How do I get started with {primary}?",
      "answer": "Begin with the fundamentals, set a clear goal, follow a structured plan, and track your progress weekly.",
    },
    {
      "question": f"How long until I see results with {primary}?",
      "answer": "Most people notice meaningful progress within a few weeks when they apply the steps consistently.",
    },
    {
      "question": f"Who benefits most from learning about {primary}?",
      "answer": f"Beginners, enthusiasts, and anyone who wants practical guidance on {topic} without unnecessary complexity.",
    },
  ]


_FAQ_TITLE_PATTERN = r"^##\s+(?:frequently asked questions|faqs?|सामान्य प्रश्न|સામાન્ય પ્રશ્નો|preguntas frecuentes|foire aux questions|häufig gestellte fragen|सतत विचारले जाणारे प्रश्न|질문|よくある質問|常见问题|أسئلة شائعة)\s*$"


def extract_outline_from_body(body: str) -> list[str]:
  outline: list[str] = []
  for line in (body or "").split("\n"):
    m = re.match(r"^##\s+(.+)$", line.strip())
    if m:
      title = re.sub(r"[*_`]", "", m.group(1)).strip()
      if not re.match(_FAQ_TITLE_PATTERN, line.strip(), re.I) and title.lower() not in ("frequently asked questions", "faq", "faqs"):
        outline.append(title)
  return outline


def extract_faqs_from_body(body: str) -> list[dict[str, str]]:
  faqs: list[dict[str, str]] = []
  in_faq = False
  current_q = ""
  for line in (body or "").split("\n"):
    stripped = line.strip()
    if re.match(_FAQ_TITLE_PATTERN, stripped, re.I):
      in_faq = True
      continue
    if not in_faq:
      continue
    if stripped.startswith("### "):
      if current_q:
        faqs.append({"question": current_q, "answer": ""})
      current_q = stripped[4:].strip()
    elif stripped and current_q:
      if faqs and faqs[-1]["question"] == current_q and not faqs[-1]["answer"]:
        faqs[-1]["answer"] = stripped
      elif faqs and faqs[-1]["question"] == current_q:
        faqs[-1]["answer"] += " " + stripped
      else:
        faqs.append({"question": current_q, "answer": stripped})
  if current_q and (not faqs or faqs[-1]["question"] != current_q):
    faqs.append({"question": current_q, "answer": ""})
  return [f for f in faqs if f.get("question")]


def strip_faq_section(body: str) -> str:
  """Remove FAQ block from article body when FAQs are returned separately."""
  lines = (body or "").split("\n")
  out: list[str] = []
  skip = False
  for line in lines:
    if re.match(_FAQ_TITLE_PATTERN, line.strip(), re.I):
      skip = True
      continue
    if skip and line.strip().startswith("## "):
      skip = False
    if not skip:
      out.append(line)
  return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def quality_report(title: str, meta: str, content: str, keywords: list[str]) -> dict[str, Any]:
  issues: list[str] = []
  score = 100
  if len(title) < 20:
    issues.append("title_short"); score -= 10
  if len(meta) < 50:
    issues.append("meta_short"); score -= 10
  if len(meta) > 160:
    issues.append("meta_long"); score -= 5
  wc = len(content.split())
  if wc < 120:
    issues.append("content_short"); score -= 20
  if not content.count("##"):
    issues.append("missing_h2"); score -= 15
  if keywords:
    primary = keywords[0].lower()
    if primary not in (title + content).lower():
      issues.append("primary_keyword_missing"); score -= 15
  return {
    "seo_score": max(0, min(100, score)),
    "seo_ready": score >= 70,
    "issues": issues,
    "heading_count": content.count("##"),
    "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
  }
