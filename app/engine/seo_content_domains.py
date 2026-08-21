"""Topic-domain content packs — real articles, not generic SEO boilerplate.

Detects fitness, health, tech, food, etc. and generates varied, topic-relevant copy.
100% custom templates — no GPT/Claude/Gemini.
"""

from __future__ import annotations

import random
import re
import secrets
import time
from typing import Any

_DOMAIN_SIGNALS: dict[str, list[str]] = {
  "fitness": [
    "workout", "exercise", "fitness", "gym", "yoga", "muscle", "cardio", "strength",
    "bodyweight", "training", "stretch", "warm-up", "warmup", "beginner workout",
    "home workout", "at-home", "weight loss", "calories", "reps", "sets",
  ],
  "health": [
    "health", "wellness", "nutrition", "diet", "sleep", "mental health", "stress",
    "meditation", "vitamin", "immune", "hydration", "healthy lifestyle", "medical",
  ],
  "coffee": [
    "coffee", "espresso", "brew", "grinder", "beans", "roast", "barista", "caffeine",
    "french press", "pour over", "latte", "cappuccino", "roastery", "arabica",
  ],
  "finance": [
    "finance", "investment", "budget", "crypto", "stocks", "savings", "banking",
    "credit", "loans", "real estate", "wealth", "asset", "tax", "portfolio",
  ],
  "ecommerce": [
    "product", "store", "buy", "review", "shop", "ecommerce", "discount", "price",
    "shipping", "deals", "best price", "checkout",
  ],
  "food": [
    "recipe", "cooking", "meal", "breakfast", "dinner", "baking", "cuisine", "food",
    "ingredient", "kitchen",
  ],
  "tech": [
    "software", "programming", "code", "app", "flutter", "python", "javascript",
    "api", "cloud", "ai", "machine learning", "developer", "web", "mobile",
  ],
  "business": [
    "marketing", "sales", "startup", "ecommerce", "seo", "email marketing",
    "conversion", "branding", "advertising", "lead", "revenue",
  ],
  "travel": [
    "travel", "hotel", "flight", "destination", "tourism", "vacation", "itinerary",
  ],
  "education": [
    "learn", "course", "study", "exam", "student", "tutorial", "lesson", "skill",
  ],
}

def extract_short_subject(text: str, max_words: int = 4) -> str:
  """Extract a clean, non-truncated short subject string respecting multi-byte Unicode/Devanagari."""
  if not text:
    return ""
  t = re.sub(r"^(?:how\s+to\s+|guide\s+to\s+|complete\s+guide\s+on\s+)", "", text, flags=re.IGNORECASE).strip()
  t = re.split(r":|\s+[-—–]\s+", t)[0].strip()
  words = t.split()
  if not words:
    return text[:30]
  if len(words) <= max_words:
    return " ".join(words)
  return " ".join(words[:max_words])


_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "hindi": "hi", "spanish": "es", "french": "fr", "german": "de",
  "portuguese": "pt", "arabic": "ar", "japanese": "ja", "chinese": "zh", "korean": "ko",
  "italian": "it", "russian": "ru", "bengali": "bn", "tamil": "ta", "marathi": "mr",
  "urdu": "ur", "vietnamese": "vi", "thai": "th", "dutch": "nl", "polish": "pl",
  "turkish": "tr", "indonesian": "id",
}


def detect_language(topic: str, keywords: list[str] | None = None, language: str | None = None) -> str:
  """Detect language code (bcp47/2-letter) from explicit parameter or script/vocabulary analysis."""
  if language and language.strip().lower() not in ("auto", "en", "english"):
    l_clean = language.strip().lower()
    return _LANG_TO_BCP47.get(l_clean, l_clean[:2])

  text = f"{topic or ''} {' '.join(keywords or [])}"

  # Devanagari script (Hindi / Marathi / Nepali)
  if re.search(r"[\u0900-\u097F]", text):
    if any(w in text for w in ("आणि", "आहे", "करतात", "येतात", "झाले", "होते", "महाराष्ट्र", "मध्ये", "कसा", "करावा", "नवीन", "व्यवसाय", "पुणे", "काय", "करणे", "आहेत", "होता", "होती")):
      return "mr"
    return "hi"

  # Gujarati script
  if re.search(r"[\u0A80-\u0AFF]", text):
    return "gu"

  # Gurmukhi script (Punjabi)
  if re.search(r"[\u0A00-\u0A7F]", text):
    return "pa"

  # Bengali script
  if re.search(r"[\u0980-\u09FF]", text):
    return "bn"

  # Tamil script
  if re.search(r"[\u0B80-\u0BFF]", text):
    return "ta"

  # Telugu script
  if re.search(r"[\u0C00-\u0C7F]", text):
    return "te"

  # Kannada script
  if re.search(r"[\u0C80-\u0CFF]", text):
    return "kn"

  # Malayalam script
  if re.search(r"[\u0D00-\u0D7F]", text):
    return "ml"

  # Urdu / Arabic / Persian script
  if re.search(r"[\u0600-\u06FF]", text):
    if any(w in text for w in ("کی", "ہے", "کو", "میں", "اور", "سے")):
      return "ur"
    return "ar"

  # Japanese (Hiragana / Katakana)
  if re.search(r"[\u3040-\u30FF]", text):
    return "ja"

  # Chinese (Hanzi)
  if re.search(r"[\u4E00-\u9FFF]", text):
    return "zh"

  # Korean (Hangul)
  if re.search(r"[\uAC00-\uD7AF]", text):
    return "ko"

  # Cyrillic (Russian)
  if re.search(r"[\u0400-\u04FF]", text):
    return "ru"

  # Thai
  if re.search(r"[\u0E00-\u0E7F]", text):
    return "th"

  # Spanish signals
  if re.search(r"[¿¡]", text) or any(w in text.lower().split() for w in ("el", "la", "los", "las", "para", "como", "cómo", "sobre", "guia", "guía", "hacer", "casa", "de", "en", "un", "una", "mejores", "consejos")):
    return "es"

  # French signals
  if any(w in text.lower().split() for w in ("le", "la", "les", "pour", "comment", "dans", "avec", "guide", "apprendre", "un", "une", "des")):
    return "fr"

  # German signals
  if "ß" in text or any(w in text.lower().split() for w in ("der", "die", "das", "und", "für", "wie", "leitfaden", "tipps", "auto", "mieten", "in", "deutschland")):
    return "de"

  # Portuguese signals
  if any(w in text.lower().split() for w in ("como", "fazer", "para", "com", "uma", "um", "guia", "dicas")):
    return "pt"

  # Vietnamese signals
  if re.search(r"[đăơưằắẳẵặầấẩẫậềếểễệồốổỗộờớởỡợừứửữự]", text, re.IGNORECASE):
    return "vi"

  return "en"


_LOCALIZED_HEADINGS: dict[str, dict[str, str]] = {
  "gu": {
    "intro": "પ્રસ્તાવના", "what_is": "શું છે", "benefits": "લાભ અને ઉપયોગિતા",
    "how_it_works": "કેવી રીતે કામ કરે છે", "key_terms": "મુખ્ય શબ્દો", "conclusion": "નિષ્કર્ષ",
    "faqs": "સામાન્ય પ્રશ્નો", "tools": "જરૂરી ટૂલ્સ", "pitfalls": "સામાન્ય ભૂલો",
    "metrics": "સફળતાના માપદંડ", "guide": "માર્ગદર્શિકા", "deeper_dive": "ઊંડાણપૂર્વક",
    "overview": "નિષ્ણાત સમીક્ષા", "quick_answer": "ઝડપી જવાબ",
    "key_takeaways": "મુખ્ય મુદ્દાઓ અને સારાંશ", "core_focus": "મુખ્ય ફોકસ", "primary_takeaway": "મુખ્ય તારણ",
    "table_of_contents": "અનુક્રમણિકા", "related_guides": "સંબંધિત માર્ગદર્શિકાઓ", "sources_credibility": "સ્ત્રોતો અને વિશ્વસનીયતા", "locations": "સેવા વિસ્તારો",
  },
  "pa": {
    "intro": "ਜਾਣ-ਪਛਾਣ", "what_is": "ਕੀ ਹੈ", "benefits": "ਲਾਭ ਅਤੇ ਵਰਤੋਂ",
    "how_it_works": "ਇਹ ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ", "key_terms": "ਮੁੱਖ ਸ਼ਬਦ", "conclusion": "ਸਿੱਟਾ",
    "faqs": "ਅਕਸર ਪੁੱਛੇ ਜਾਂਦੇ ਸਵਾਲ", "tools": "ਜ਼ਰੂਰੀ ਟੂਲ", "pitfalls": "ਆਮ ਗਲਤੀਆਂ",
    "metrics": "ਸਫਲਤਾ ਦੇ ਮਾਪਦੰਡ", "guide": "ਗਾਈਡ", "deeper_dive": "ਵਿਸਥਾਰ ਨਾਲ",
    "overview": "ਮਾਹਰ ਸਮੀਖਿਆ", "quick_answer": "ਤੁਰੰਤ ਜਵਾਬ",
    "key_takeaways": "ਮੁੱਖ ਬਿੰਦੂ ਅਤੇ ਸਾਰ", "core_focus": "ਮੁੱਖ ਫੋਕਸ", "primary_takeaway": "ਮੁੱਖ ਸਿੱਟਾ",
    "table_of_contents": "ਵਿਸ਼ਾ સૂਚੀ", "related_guides": "સંબੰਧਿਤ ਗਾਈਡਾਂ", "sources_credibility": "ਸਰੋਤ এবং ਵਿਸ਼ਵਾਸਯੋਗਤਾ", "locations": "ਸੇਵਾ ਖੇਤਰ",
  },
  "te": {
    "intro": "పరిచయం", "what_is": "అంటే ఏమిటి", "benefits": "ప్రయోజనాలు మరియు ఉపయోగాలు",
    "how_it_works": "ఇది ఎలా పనిచేస్తుంది", "key_terms": "ముఖ్యమైన పదాలు", "conclusion": "ముగింపు",
    "faqs": "తరచుగా అడిగే ప్రశ్నలు", "tools": "అవసరమైన టూల్స్", "pitfalls": "సాధారణ పొరపాట్లు",
    "metrics": "విజయం కొలమానాలు", "guide": "మార్గదర్శిని", "deeper_dive": "వివరంగా",
    "overview": "నిపుణుల పరిశీలన", "quick_answer": "త్వరిత సమాధానం",
    "key_takeaways": "ముఖ్య అంశాలు మరియు సారాంశం", "core_focus": "ప్రధాన దృష్టి", "primary_takeaway": "ప్రధాన ముగింపు",
    "table_of_contents": "విషయసూచిక", "related_guides": "సంబంధిత మార్గదర్శకాలు", "sources_credibility": "మూలాలు మరియు విశ్వసనీయత", "locations": "సేవా ప్రాంతాలు",
  },
  "kn": {
    "intro": "పరిచయ", "what_is": "ಎಂದರೆ ఏమిటి", "benefits": "ప్రయోజనగళు మత్తు ఉపయోగిసబహుదు",
    "how_it_works": "ఇదు హేగె కెలస మాడుత్తదె", "key_terms": "ముఖ్య పదగళు", "conclusion": "ముగింపు",
    "faqs": "సామాన్య ప్రశ్నెగళు", "tools": "అగత్య టూల్స్", "pitfalls": "సామాన్య తప్పుగళు",
    "metrics": "యశస్సిన ప్రమాణగళు", "guide": "మార్గదర్శి", "deeper_dive": "వివరవాగి",
    "overview": "నిపుణర వివరణె", "quick_answer": "త్వరిత ఉత్తర",
    "key_takeaways": "ముఖ్య అంశగళు మత్తు సారాంశ", "core_focus": "ప్రధాన గమన", "primary_takeaway": "ముఖ్య నిష్కర్షె",
    "table_of_contents": "విషయసూచిక", "related_guides": "సంబంధిత మార్గదర్శకాలు", "sources_credibility": "మూలాలు", "locations": "సేవా ప్రాంతాలు",
  },
  "ml": {
    "intro": "ആമുഖം", "what_is": "എന്നാൽ എന്താണ്", "benefits": "നേട്ടങ്ങളും ഉപയോഗങ്ങളും",
    "how_it_works": "ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു", "key_terms": "പ്രധാന പദങ്ങൾ", "conclusion": "ഉപസംഹാരം",
    "faqs": "ചോദ്യങ്ങൾ", "tools": "ആവശ്യമായ ടൂളുകൾ", "pitfalls": "തെറ്റുകൾ",
    "metrics": "വിജയ മാനദണ്ഡങ്ങൾ", "guide": "വഴികാട്ടി", "deeper_dive": "വിശദമായി",
    "overview": "വിദഗ്ദ്ധ അവലോകനം", "quick_answer": "പെട്ടെന്നുള്ള മറുപടി",
    "key_takeaways": "പ്രധാന പോയിന്റുകൾ", "core_focus": "പ്രധാന ശ്രദ്ധ", "primary_takeaway": "പ്രധാന കണ്ടെത്തൽ",
    "table_of_contents": "ഉള്ളടക്കം", "related_guides": "ബന്ധപ്പെട്ട മാർഗ്ഗനിർദ്ദേശങ്ങൾ", "sources_credibility": "ഉറവിടങ്ങൾ", "locations": "സേവന മേഖലകൾ",
  },
  "hi": {
    "intro": "परिचय", "what_is": "क्या है", "benefits": "लाभ और उपयोग",
    "how_it_works": "यह कैसे काम करता है", "key_terms": "मुख्य शब्द", "conclusion": "निष्कर्ष",
    "faqs": "सामान्य प्रश्न", "tools": "आवश्यक टूल्स", "pitfalls": "सामान्य गलतियाँ",
    "metrics": "सफलता के पैमाने", "guide": "चरण-दर-चरण मार्गदर्शिका", "deeper_dive": "गहराई से जानें",
    "overview": "विशेषज्ञ अवलोकन", "quick_answer": "त्वरित उत्तर",
    "key_takeaways": "मुख्य बिंदु और कार्यकारी सारांश", "core_focus": "मुख्य केंद्र", "primary_takeaway": "प्राथमिक सीख",
    "table_of_contents": "अनुक्रमणिका", "related_guides": "संबंधित मार्गदर्शिकाएँ", "sources_credibility": "स्रोत और विश्वसनीयता", "locations": "सेवा क्षेत्र",
  },
  "es": {
    "intro": "Introducción", "what_is": "¿Qué es", "benefits": "Beneficios y Casos de Uso",
    "how_it_works": "Cómo Funciona", "key_terms": "Términos Clave", "conclusion": "Conclusión",
    "faqs": "Preguntas Frecuentes", "tools": "Herramientas Necesarias", "pitfalls": "Errores Comunes",
    "metrics": "Métricas de Éxito", "guide": "Guía Paso a Paso", "deeper_dive": "Análisis Detallado",
    "overview": "Visión General Experta", "quick_answer": "Respuesta Rápida",
    "key_takeaways": "Puntos Clave y Resumen Ejecutivo", "core_focus": "Enfoque Principal", "primary_takeaway": "Conclusión Principal",
    "table_of_contents": "Tabla de contenidos", "related_guides": "Guías relacionadas", "sources_credibility": "Fuentes y credibilidad", "locations": "Áreas de servicio",
  },
  "de": {
    "intro": "Einführung", "what_is": "Was ist", "benefits": "Vorteile und Anwendungsfälle",
    "how_it_works": "Wie es funktioniert", "key_terms": "Schlüsselbegriffe", "conclusion": "Fazit",
    "faqs": "Häufig gestellte Fragen", "tools": "Wichtige Werkzeuge", "pitfalls": "Häufige Fehler",
    "metrics": "Erfolgsfaktoren", "guide": "Schritt-für-Schritt-Anleitung", "deeper_dive": "Vertiefung",
    "overview": "Experten-Übersicht", "quick_answer": "Schnelle Antwort",
    "key_takeaways": "Wichtigste Erkenntnisse & Zusammenfassung", "core_focus": "Hauptfokus", "primary_takeaway": "Kernaussage",
    "table_of_contents": "Inhaltsverzeichnis", "related_guides": "Verwandte Leitfäden", "sources_credibility": "Quellen & Glaubwürdigkeit", "locations": "Servicebereiche",
  },
  "fr": {
    "intro": "Introduction", "what_is": "Qu'est-ce que", "benefits": "Avantages et Cas d'Utilisation",
    "how_it_works": "Comment ça marche", "key_terms": "Termes Clés", "conclusion": "Conclusion",
    "faqs": "Foire Aux Questions", "tools": "Outils Essentiels", "pitfalls": "Erreurs Courantes",
    "metrics": "Indicateurs de Succès", "guide": "Guide Étape par Étape", "deeper_dive": "Analyse Approfondie",
    "overview": "Aperçu d'Expert", "quick_answer": "Réponse Rapide",
    "key_takeaways": "Points Clés et Résumé Exécutif", "core_focus": "Focus Principal", "primary_takeaway": "Enseignement Principal",
    "table_of_contents": "Table des matières", "related_guides": "Guides connexes", "sources_credibility": "Sources et crédibilité", "locations": "Zones de service",
  },
  "mr": {
    "intro": "परिचय", "what_is": "म्हणजे काय", "benefits": "फायदे आणि उपयोग",
    "how_it_works": "हे कसे कार्य करते", "key_terms": "महत्वाचे शब्द", "conclusion": "निष्कर्ष",
    "faqs": "सतत विचारले जाणारे प्रश्न", "tools": "आवश्यक साधने", "pitfalls": "सामान्य चुका",
    "metrics": "यशाचे निकष", "guide": "मार्गदर्शिका", "deeper_dive": "सविस्तर माहिती",
    "overview": "तज्ञांचे पुनरावलोकन", "quick_answer": "थोडक्यात उत्तर",
    "key_takeaways": "महत्वाचे मुद्दे", "core_focus": "मुख्य लक्ष", "primary_takeaway": "प्राथमिक निष्कर्ष",
    "table_of_contents": "अनुक्रमणिका", "related_guides": "संबंधित मार्गदर्शिका", "sources_credibility": "स्रोत आणि विश्वासार्हता", "locations": "सेवा क्षेत्र",
  },
  "bn": {
    "intro": "ভূমিকা", "what_is": "কী এবং কেন", "benefits": "সুবিধা এবং ব্যবহার",
    "how_it_works": "কীভাবে কাজ করে", "key_terms": "মূল শব্দাবলী", "conclusion": "উপসংহার",
    "faqs": "সাধারণ প্রশ্নাবলী", "tools": "প্রয়োজনীয় টুলস", "pitfalls": "সাধারণ ভুলসমূহ",
    "metrics": "সফলতার সূচক", "guide": "ধাপে ধাপে নির্দেশিকা", "deeper_dive": "বিস্তারিত বিশ্লেষণ",
    "overview": "বিশেষজ্ঞ পর্যালোচনা", "quick_answer": "সংক্ষিপ্ত উত্তর",
    "key_takeaways": "প্রধান সারসংক্ষেপ", "core_focus": "মূল ফোকাস", "primary_takeaway": "প্রধান শিক্ষা",
  },
  "ta": {
    "intro": "அறிமுகம்", "what_is": "என்றால் என்ன", "benefits": "நன்மைகள் மற்றும் பயன்பாடுகள்",
    "how_it_works": "எவ்வாறு செயல்படுகிறது", "key_terms": "முக்கிய சொற்கள்", "conclusion": "முடிவுரை",
    "faqs": "அடிக்கடி கேட்கப்படும் கேள்விகள்", "tools": "தேவையான கருவிகள்", "pitfalls": "பொதுவான தவறுகள்",
    "metrics": "வெற்றி அளவீடுகள்", "guide": "படிப் படியான வழிகாட்டி", "deeper_dive": "ஆழ்ந்த பகுப்பாய்வு",
    "overview": "வல்லுனர் கண்ணோட்டம்", "quick_answer": "விரைவான பதில்",
    "key_takeaways": "முக்கிய அம்சங்கள்", "core_focus": "முதன்மை நோக்கம்", "primary_takeaway": "முக்கிய முடிவு",
  },
  "ur": {
    "intro": "تعارف", "what_is": "کیا ہے", "benefits": "فوائد اور استعمال",
    "how_it_works": "یہ کیسے کام کرتا ہے", "key_terms": "اہم اصطلاحات", "conclusion": "نتیجہ",
    "faqs": "عام سوالات", "tools": "ضروری ٹولز", "pitfalls": "عام غلطیاں",
    "metrics": "کامیابی کے پیمانے", "guide": "مکمل رہنما", "deeper_dive": "تفصیلی جائزہ",
    "overview": "ماہرانہ جائزہ", "quick_answer": "فوری جواب",
    "key_takeaways": "اہم نکات اور خلاصہ", "core_focus": "بنیادی توجہ", "primary_takeaway": "بنیادی حاصل",
  },
  "ar": {
    "intro": "مقدمة", "what_is": "ما هو", "benefits": "الفوائد وحالات الاستخدام",
    "how_it_works": "كيف يعمل", "key_terms": "المصطلحات الرئيسية", "conclusion": "الخاتمة",
    "faqs": "الأسئلة الشائعة", "tools": "الأدوات المطلوبة", "pitfalls": "الأخطاء الشائعة",
    "metrics": "مقاييس النجاح", "guide": "دليل خطوة بخطوة", "deeper_dive": "تحليل عميق",
    "overview": "نظرة عامة من الخبراء", "quick_answer": "إجابة سريعة",
    "key_takeaways": "النقاط الرئيسية والملخص التنفيذي", "core_focus": "التركيز الأساسي", "primary_takeaway": "النتيجة الرئيسية",
  },
  "ja": {
    "intro": "はじめに", "what_is": "とは何か", "benefits": "メリットと活用事例",
    "how_it_works": "仕組みと手順", "key_terms": "重要用語", "conclusion": "まとめ",
    "faqs": "よくある質問", "tools": "必要なツール", "pitfalls": "注意すべき点",
    "metrics": "成功指標", "guide": "ステップバイステップガイド", "deeper_dive": "詳細分析",
    "overview": "専門家による概要", "quick_answer": "要約回答",
    "key_takeaways": "主要なポイントと要約", "core_focus": "主な焦点", "primary_takeaway": "最大の学び",
  },
  "zh": {
    "intro": "引言", "what_is": "什么是", "benefits": "优势与应用场景",
    "how_it_works": "工作原理与步骤", "key_terms": "核心术语", "conclusion": "总结",
    "faqs": "常见问题解答", "tools": "所需工具", "pitfalls": "常见误区",
    "metrics": "成功指标", "guide": "逐步指南", "deeper_dive": "深度解析",
    "overview": "专家概述", "quick_answer": "快速解答",
    "key_takeaways": "核心要点与执行摘要", "core_focus": "核心重点", "primary_takeaway": "主要收获",
  },
  "ko": {
    "intro": "서론", "what_is": "이란 무엇인가", "benefits": "혜택 및 활용 사례",
    "how_it_works": "작동 방식", "key_terms": "핵심 용어", "conclusion": "결론",
    "faqs": "자주 묻는 질문", "tools": "필수 도구", "pitfalls": "주의할 점",
    "metrics": "성공 지표", "guide": "단계별 가이드", "deeper_dive": "심층 분석",
    "overview": "전문가 개요", "quick_answer": "빠른 요약",
    "key_takeaways": "핵심 요약 및 실행 요약", "core_focus": "핵심 초점", "primary_takeaway": "주요 시사점",
  },
  "ru": {
    "intro": "Введение", "what_is": "Что такое", "benefits": "Преимущества и варианты использования",
    "how_it_works": "Как это работает", "key_terms": "Ключевые термины", "conclusion": "Заключение",
    "faqs": "Часто задаваемые вопросы", "tools": "Необходимые инструменты", "pitfalls": "Распространенные ошибки",
    "metrics": "Метрики успеха", "guide": "Пошаговое руководство", "deeper_dive": "Глубокий анализ",
    "overview": "Экспертный обзор", "quick_answer": "Краткий ответ",
    "key_takeaways": "Ключевые выводы и резюме", "core_focus": "Основной фокус", "primary_takeaway": "Главный вывод",
  },
  "pt": {
    "intro": "Introdução", "what_is": "O que é", "benefits": "Benefícios e Casos de Uso",
    "how_it_works": "Como Funciona", "key_terms": "Termos Chave", "conclusion": "Conclusão",
    "faqs": "Perguntas Frequentes", "tools": "Ferramentas Necessárias", "pitfalls": "Erros Comuns",
    "metrics": "Métricas de Sucesso", "guide": "Guia Passo a Passo", "deeper_dive": "Análise Detalhada",
    "overview": "Visão Geral Especializada", "quick_answer": "Resposta Rápida",
    "key_takeaways": "Principais Conclusões e Resumo Executivo", "core_focus": "Foco Principal", "primary_takeaway": "Principal Conclusão",
  },
  "it": {
    "intro": "Introduzione", "what_is": "Che cos'è", "benefits": "Vantaggi e Casi d'Uso",
    "how_it_works": "Come Funziona", "key_terms": "Termini Chiave", "conclusion": "Conclusione",
    "faqs": "Domande Frequenti", "tools": "Strumenti Essenziali", "pitfalls": "Errori Comuni",
    "metrics": "Metriche di Successo", "guide": "Guida Passo Passo", "deeper_dive": "Analisi Approfondita",
    "overview": "Panoramica dell'Esperto", "quick_answer": "Risposta Rapida",
    "key_takeaways": "Punti Chiave e Sintesi Esecutiva", "core_focus": "Focus Principale", "primary_takeaway": "Conclusione Principale",
  },
}


def get_localized_heading(key: str, lang: str | None = "en") -> str:
  code = (lang or "en").lower()[:2]
  loc = _LOCALIZED_HEADINGS.get(code, {})
  if key in loc:
    return loc[key]
  fallback_map = {
    "intro": "Introduction", "what_is": "What Is", "benefits": "Benefits and Use Cases",
    "how_it_works": "How It Works", "key_terms": "Key Terms", "conclusion": "Conclusion",
    "faqs": "FAQs", "tools": "Tools Involved", "pitfalls": "Common Pitfalls",
    "metrics": "Success Metrics", "guide": "Step-by-Step Guide", "deeper_dive": "Deeper dive",
    "overview": "Expert Overview", "quick_answer": "Quick Answer",
    "key_takeaways": "Key Takeaways & Executive Summary",
    "core_focus": "Core Focus", "primary_takeaway": "Primary Takeaway",
  }
  return fallback_map.get(key, key.title())


_DYNAMIC_STARTERS = [
  "Understanding", "Mastering", "Evaluating", "Navigating", "Exploring",
  "Optimizing", "Implementing", "Leveraging", "Streamlining", "Building",
]


def dynamic_sentence_starter(seed: int, topic: str) -> str:
  starter = _DYNAMIC_STARTERS[seed % len(_DYNAMIC_STARTERS)]
  return f"{starter} {topic} effectively requires a clear roadmap and data-driven execution."


_SECONDARY_SUGGESTIONS: dict[str, list[str]] = {
  "fitness": [
    "beginner workout plan", "home exercise routine", "bodyweight exercises",
    "at-home fitness", "beginner fitness routine", "daily workout plan",
    "strength training at home", "no equipment workout", "healthy lifestyle",
  ],
  "health": [
    "wellness tips", "healthy habits", "nutrition guide", "self-care routine",
    "mental wellness", "daily health routine",
  ],
  "food": [
    "easy recipes", "meal prep", "cooking tips", "ingredient list", "healthy meals",
  ],
  "tech": [
    "best practices", "step-by-step guide", "tutorial", "tips and tricks", "setup guide",
  ],
  "business": [
    "best practices", "strategy guide", "tips for beginners", "growth tactics",
  ],
  "travel": [
    "travel tips", "packing list", "budget travel", "things to do", "local guide",
  ],
  "education": [
    "study tips", "learning path", "beginner guide", "practice exercises",
  ],
  "enterprise": [
    "ERP modules", "implementation guide", "vendor comparison", "ROI calculator",
    "inventory tracking", "financial reporting", "manufacturing workflow", "cloud deployment",
    "data migration", "user training",
  ],
  "general": [
    "beginner guide", "step-by-step", "best practices", "tips and tricks",
    "how to get started", "common mistakes",
  ],
  "hi_general": [
    "शुरुआती गाइड", "चरण-दर-चरण मार्गदर्शिका", "सर्वोत्तम कार्यप्रणालियाँ", "व्यावहारिक टिप्स",
    "सफलता के उपाय", "विशेषज्ञ सलाह",
  ],
  "es_general": [
    "guía para principiantes", "paso a paso", "mejores prácticas", "consejos y trucos",
  ],
}


def make_variation_seed(explicit: int | None = None) -> int:
  if explicit is not None:
    return int(explicit) & 0x7FFFFFFF
  return secrets.randbits(31)


def _pick(seed: int, options: list[str]) -> str:
  if not options:
    return ""
  return options[seed % len(options)]


def detect_domain(topic: str, keywords: list[str]) -> str:
  text = f"{topic} {' '.join(keywords)}".lower()
  scores = {d: sum(1 for s in sigs if s in text) for d, sigs in _DOMAIN_SIGNALS.items()}
  best = max(scores, key=scores.get)
  return best if scores[best] > 0 else "general"


def expand_keywords(topic: str, keywords: list[str], domain: str) -> dict[str, Any]:
  primary = (keywords[0] if keywords else topic).strip()
  secondary: list[str] = []
  seen = {primary.lower()}
  for kw in keywords[1:]:
    k = kw.strip()
    if k and k.lower() not in seen:
      secondary.append(k)
      seen.add(k.lower())
  pool = _SECONDARY_SUGGESTIONS.get(domain, _SECONDARY_SUGGESTIONS["general"])
  for s in pool:
    if s.lower() not in seen and len(secondary) < 9:
      secondary.append(s)
      seen.add(s.lower())
  return {"primary": primary, "secondary": secondary}


def build_structured_outline(
  topic: str,
  primary: str,
  *,
  domain: str,
  category: str,
  seed: int,
) -> list[dict[str, str]]:
  lang_h1 = detect_language(topic, [primary])
  if lang_h1 != "en":
    from app.engine import seo_content_engine
    h1 = topic
    outlines_str = seo_content_engine.build_outline(topic, [primary], category, language=lang_h1)
    return [{"level": "h1", "text": h1}] + [{"level": "h2", "text": t} for t in outlines_str]
  if domain == "fitness":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"Benefits of {primary.title()} for Beginners"},
      {"level": "h2", "text": "Essential Tips Before Starting"},
      {"level": "h2", "text": f"Beginner {primary.title()} Plan"},
      {"level": "h3", "text": "Warm-Up Exercises"},
      {"level": "h3", "text": "Upper Body Exercises"},
      {"level": "h3", "text": "Lower Body Exercises"},
      {"level": "h3", "text": "Core Exercises"},
      {"level": "h3", "text": "Cool-Down and Stretching"},
      {"level": "h2", "text": "Weekly Workout Schedule"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "How to Stay Consistent"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if domain == "enterprise":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"What Is {primary.title()}?"},
      {"level": "h2", "text": "Core Modules and Features"},
      {"level": "h3", "text": "Finance and Accounting"},
      {"level": "h3", "text": "Inventory and Supply Chain"},
      {"level": "h3", "text": "Manufacturing and Operations"},
      {"level": "h3", "text": "HR and Payroll"},
      {"level": "h2", "text": f"Benefits of {primary.title()} for Businesses"},
      {"level": "h2", "text": "How to Choose the Right Solution"},
      {"level": "h2", "text": "Implementation Roadmap"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if category == "how_to_guide":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": "What You Need Before Starting"},
      {"level": "h2", "text": "Step-by-Step Instructions"},
      {"level": "h3", "text": "Step 1: Prepare"},
      {"level": "h3", "text": "Step 2: Execute"},
      {"level": "h3", "text": "Step 3: Refine"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "Conclusion"},
    ]
  return [
    {"level": "h1", "text": h1},
    {"level": "h2", "text": "Introduction"},
    {"level": "h2", "text": f"Why {primary.title()} Matters"},
    {"level": "h2", "text": f"Key Benefits of {primary.title()}"},
    {"level": "h2", "text": "Practical Tips and Best Practices"},
    {"level": "h2", "text": f"How to Get Started With {primary.title()}"},
    {"level": "h2", "text": "Common Mistakes to Avoid"},
    {"level": "h2", "text": "Conclusion"},
  ]


def _fitness_article(topic: str, primary: str, tone: str, seed: int) -> tuple[str, str, str]:
  title = _pick(seed, [
    f"{topic}: A Professional Beginner Workout Plan for Better Fitness",
    f"{primary.title()} for Beginners: Complete Home Fitness Guide",
    f"{topic} — Step-by-Step Plan for Strength and Endurance",
  ])
  meta = _trim(
    f"Discover an effective {primary} with a step-by-step beginner workout plan. "
    "Learn essential exercises, weekly schedules, and tips to build strength "
    "and improve overall fitness from home."
  )
  intro = _pick(seed + 1, [
    (
      f"Starting a fitness journey can seem challenging, especially for beginners. "
      f"However, a structured **{primary}** routine provides a convenient and affordable way "
      "to improve overall health without a gym membership. With consistency and the right exercises, "
      "beginners can develop strength, flexibility, and endurance from home."
    ),
    (
      f"A **{primary}** is one of the most accessible ways to begin improving your health. "
      "No expensive equipment is required — just space, commitment, and a clear plan. "
      "This guide walks you through everything a beginner needs to train safely and effectively."
    ),
  ])
  body = f"""# {title}

## Introduction

{intro}

## Benefits of {primary.title()} for Beginners

Home workouts offer flexibility and convenience. They eliminate travel time, require little to no equipment, and fit any schedule. Regular activity improves cardiovascular health, boosts energy, supports weight management, and reduces stress.

## Essential Tips Before Starting

Before beginning, remember to:

- Set realistic fitness goals
- Start slowly and focus on proper form
- Stay hydrated throughout the day
- Wear comfortable clothing and supportive footwear
- Always include warm-up and cool-down sessions

## Beginner {primary.title()} Plan

### Warm-Up Exercises

Begin with five minutes of light cardio: marching in place, arm circles, and jumping jacks prepare your body for exercise.

### Upper Body Exercises

- **Push-ups:** 3 sets of 10 repetitions
- **Incline push-ups:** 3 sets of 12 repetitions

### Lower Body Exercises

- **Squats:** 3 sets of 15 repetitions
- **Lunges:** 3 sets of 12 repetitions per leg

### Core Exercises

- **Plank:** Hold 30 seconds, 3 rounds
- **Bicycle crunches:** 3 sets of 20 repetitions

### Cool-Down and Stretching

Spend five to ten minutes stretching major muscle groups to improve flexibility and reduce soreness.

## Weekly Workout Schedule

| Day | Activity |
|-----|----------|
| Monday | Full body workout |
| Tuesday | Cardio and stretching |
| Wednesday | Upper body training |
| Thursday | Rest or walking |
| Friday | Lower body training |
| Saturday | Core exercises |
| Sunday | Recovery and stretching |

## Common Mistakes to Avoid

Many beginners overtrain or expect immediate results. Skipping warm-ups, ignoring proper technique, or lacking consistency slows progress. Focus on gradual improvement rather than perfection.

## How to Stay Consistent

Track progress, follow your weekly schedule, and maintain a balanced diet. Even 30 minutes of daily exercise leads to noticeable improvements over time.

## Conclusion

A well-designed **{primary}** routine is an excellent starting point for beginners. With discipline, patience, and gradual progress, anyone can build strength and healthier habits without leaving home.
"""
  return title, meta, body.strip()


def _enterprise_article(
  topic: str,
  primary: str,
  tone: str,
  seed: int,
  audience: str | None,
) -> tuple[str, str, str]:
  aud = f" for {audience}" if audience else ""
  title = _pick(seed, [
    f"{topic}: Complete Guide to Selection and Implementation{aud}",
    f"{primary.title()} — Features, Benefits, and Best Practices",
    f"How to Choose and Deploy {primary.title()} Successfully",
  ])
  meta = _trim(
    f"Learn what {primary} is, core modules, selection criteria, and implementation steps. "
    f"Practical guide for manufacturing, inventory, and finance teams{aud}."
  )
  intro = _pick(seed + 1, [
    (
      f"**{primary.title()}** integrates finance, inventory, manufacturing, HR, and reporting "
      "into one system so teams stop juggling spreadsheets and disconnected tools."
    ),
    (
      f"Organizations adopt **{primary}** to unify operations, improve data accuracy, "
      "and speed up decisions across departments."
    ),
  ])
  body = f"""# {title}

## Introduction

{intro}

## What Is {primary.title()}?

Enterprise Resource Planning (ERP) software connects core business processes in a single platform. Instead of isolated apps for accounting, stock, and production, ERP gives one shared database and workflow.

## Core Modules and Features

### Finance and Accounting

General ledger, accounts payable/receivable, tax compliance, and real-time financial reporting.

### Inventory and Supply Chain

Track stock levels, purchase orders, suppliers, and warehouse movements with accurate reorder alerts.

### Manufacturing and Operations

Plan production, bill of materials (BOM), work orders, and shop-floor scheduling for make-to-stock or make-to-order models.

### HR and Payroll

Employee records, attendance, payroll runs, and basic workforce analytics in the same system.

## Benefits of {primary.title()} for Businesses

- **Single source of truth** — one dataset for sales, inventory, and finance
- **Fewer manual errors** — automated postings between modules
- **Faster reporting** — dashboards for cash flow, margins, and stock aging
- **Scalable growth** — add users, warehouses, or legal entities without rebuilding processes

## How to Choose the Right Solution

1. Map your must-have modules (finance, inventory, MRP, CRM, etc.)
2. Compare cloud vs on-premise and total cost of ownership
3. Check integration with existing tools (e-commerce, POS, BI)
4. Run a pilot with real data from one department
5. Verify vendor support, training, and migration services

## Implementation Roadmap

| Phase | Focus |
|-------|--------|
| Discovery | Process mapping, gap analysis, data audit |
| Design | Workflows, roles, chart of accounts, item masters |
| Build | Configuration, integrations, test environment |
| Test | UAT, parallel run, fix reconciliations |
| Go-live | Cutover, hypercare support, training |

## Common Mistakes to Avoid

- Buying more modules than the team can adopt in year one
- Skipping data cleanup before migration
- Weak change management — users revert to spreadsheets
- No executive sponsor for cross-department decisions

## Conclusion

The right **{primary}** implementation pays off through visibility, automation, and faster closes. Start with clear goals, clean master data, and phased rollout.
"""
  return title, meta, body.strip()


def _general_article(
  topic: str,
  primary: str,
  *,
  domain: str,
  tone: str,
  seed: int,
  audience: str | None,
) -> tuple[str, str, str]:
  aud = f" for {audience}" if audience else ""
  subject = extract_short_subject(primary) if (primary and len(primary.split()) <= 4) else extract_short_subject(topic)

  title = _pick(seed, [
    f"{subject}: Practical Guide for Beginners{aud}",
    f"How to Understand and Apply {subject}{aud}",
    f"Everything You Need to Know About {subject}",
  ])
  meta = _trim(
    f"Learn about {subject} with clear, practical guidance. "
    f"Covers benefits, step-by-step tips, and expert advice{aud}."
  )
  benefit = {
    "health": "supports physical and mental well-being, builds sustainable habits, and improves daily energy",
    "food": "saves time, improves nutrition, and makes home cooking enjoyable for any skill level",
    "tech": "helps you work smarter, avoid common pitfalls, and build reliable solutions faster",
    "business": "drives measurable growth, improves customer trust, and creates long-term value",
    "travel": "helps you plan smarter trips, save money, and enjoy destinations more fully",
    "education": "accelerates learning, builds confidence, and creates lasting skills",
    "general": "saves time, reduces mistakes, and delivers practical results you can apply immediately",
  }.get(domain, "delivers practical, real-world benefits you can apply right away")

  intro_variants = [
    (
      f"Understanding **{subject}** is valuable for anyone who wants clear, actionable guidance. "
      f"This article explains what matters most, how to begin, and which mistakes to avoid."
    ),
    (
      f"Whether you are just starting out or refining your approach, **{subject}** offers real benefits. "
      f"This guide breaks the topic into simple steps anyone can follow."
    ),
  ]
  body = f"""# {title}

## Introduction

{_pick(seed + 2, intro_variants)}

## Why {subject} Matters

{subject} {benefit}. Investing time in the right approach pays off through better outcomes and fewer setbacks.

## Key Benefits of {subject}


- Practical knowledge you can apply immediately
- Clear structure that saves time and reduces confusion
- Confidence to make informed decisions
- Sustainable habits that compound over weeks and months

## Practical Tips and Best Practices

- Start with a clear goal and realistic timeline
- Learn fundamentals before advanced techniques
- Track progress and adjust based on results
- Seek reliable sources and proven methods
- Stay consistent — small daily steps beat occasional bursts

## How to Get Started With {primary.title()}

1. Research the basics and define what success looks like for you
2. Gather any tools or resources you need
3. Follow a structured plan for the first 2–4 weeks
4. Review results and refine your approach

## Common Mistakes to Avoid

- Rushing without understanding fundamentals
- Expecting instant results instead of gradual progress
- Skipping planning and jumping straight to execution
- Giving up too early before habits take hold

## Conclusion

**{primary.title()}** becomes manageable when you follow a clear plan. Apply these steps consistently, track your progress, and refine your approach over time.
"""
  return title, meta, body.strip()


def build_domain_faqs(topic: str, primary: str, domain: str, seed: int) -> list[dict[str, str]]:
  if domain == "fitness":
    variants = [
      [
        {
          "question": f"What is the best {primary} for beginners?",
          "answer": "A combination of squats, push-ups, lunges, planks, and stretching provides a balanced beginner workout.",
        },
        {
          "question": "Can beginners build muscle with home workouts?",
          "answer": "Yes. Bodyweight exercises and progressive overload help beginners build muscle and improve strength.",
        },
        {
          "question": "How long should a beginner exercise at home?",
          "answer": "Aim for 20 to 45 minutes of exercise, three to five days per week.",
        },
        {
          "question": "Do I need equipment for a home workout?",
          "answer": "No. Most beginner exercises can be performed using body weight alone.",
        },
        {
          "question": "How long does it take to see results from home workouts?",
          "answer": "Most people notice improvements in energy and fitness within four to eight weeks of consistent training.",
        },
      ],
      [
        {
          "question": f"How often should I do a {primary}?",
          "answer": "Three to five sessions per week with rest days allows muscles to recover and adapt.",
        },
        {
          "question": "Is warming up necessary for beginners?",
          "answer": "Yes. Five minutes of light movement reduces injury risk and improves performance.",
        },
        {
          "question": f"What exercises are safest for a {primary}?",
          "answer": "Bodyweight squats, incline push-ups, lunges, and planks are safe, effective starter movements.",
        },
        {
          "question": "Can I lose weight with home workouts alone?",
          "answer": "Exercise combined with a balanced diet supports weight management; consistency matters most.",
        },
      ],
    ]
    return variants[seed % len(variants)]

  if domain == "enterprise":
    return [
      {
        "question": f"What is {primary}?",
        "answer": (
          f"{primary.title()} is integrated business software that connects finance, inventory, "
          "manufacturing, HR, and reporting in one platform."
        ),
      },
      {
        "question": f"What modules are included in {primary}?",
        "answer": "Typical modules include accounting, inventory, procurement, manufacturing, CRM, and HR/payroll.",
      },
      {
        "question": f"How long does {primary} implementation take?",
        "answer": "Small businesses may go live in 3–6 months; mid-size deployments often take 6–12 months depending on scope.",
      },
      {
        "question": f"Cloud or on-premise {primary} — which is better?",
        "answer": "Cloud ERP offers faster updates and lower upfront cost; on-premise suits strict data-residency or heavy customization needs.",
      },
      {
        "question": f"Who needs {primary}?",
        "answer": "Growing companies with complex inventory, multi-location operations, or manual reconciliation pain benefit most.",
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


def get_language_name(code_or_name: str | None) -> str:
  if not code_or_name:
    return "English"
  c = code_or_name.strip().lower()
  if c in _BCP47_TO_LANG_NAME:
    return _BCP47_TO_LANG_NAME[c]
  bcp = _LANG_TO_BCP47.get(c, c[:2])
  return _BCP47_TO_LANG_NAME.get(bcp, code_or_name.strip().title())


_BCP47_TO_LANG_NAME = {
  "en": "English",
  "hi": "Hindi",
  "gu": "Gujarati",
  "mr": "Marathi",
  "pa": "Punjabi",
  "bn": "Bengali",
  "ta": "Tamil",
  "te": "Telugu",
  "kn": "Kannada",
  "ml": "Malayalam",
  "ur": "Urdu",
  "ar": "Arabic",
  "es": "Spanish",
  "fr": "French",
  "de": "German",
  "ja": "Japanese",
  "zh": "Chinese",
  "ko": "Korean",
  "ru": "Russian",
  "th": "Thai",
  "vi": "Vietnamese",
  "nl": "Dutch",
  "pl": "Polish",
  "tr": "Turkish",
  "id": "Indonesian",
  "it": "Italian",
  "pt": "Portuguese",
}


def _localized_article(
  topic: str,
  primary: str,
  *,
  domain: str,
  tone: str,
  seed: int,
  audience: str | None,
  lang: str,
  category: str = "blog_article",
) -> tuple[str, str, str]:
  intro_h = get_localized_heading("intro", lang)
  benefits_h = get_localized_heading("benefits", lang)
  how_h = get_localized_heading("how_it_works", lang)
  guide_h = get_localized_heading("guide", lang)
  pitfalls_h = get_localized_heading("pitfalls", lang)
  conclusion_h = get_localized_heading("conclusion", lang)

  if lang == "hi":
    title = f"{primary}: संपूर्ण मार्गदर्शन और उपयोग"
    meta = _trim(f"{primary} और {topic} के बारे में विस्तृत और व्यावहारिक जानकारी प्राप्त करें।")
    intro_txt = f"**{primary}** और **{topic}** के मूल सिद्धांतों को समझना और उन्हें सही तरीके से लागू करना अत्यंत महत्वपूर्ण है।"
    why_txt = f"**{primary}** का सही ज्ञान समय की बचत करता है और बेहतर परिणाम प्रदान करता है।"
    b1, b2, b3 = "व्यावहारिक ज्ञान जिसे तुरंत लागू किया जा सकता है", "स्पष्ट प्रक्रिया जो गलतियों को कम करती है", "सफलता के लिए दीर्घकालिक रणनीतियाँ"
    p1, p2 = "बिना योजना के काम शुरू करना", "परिणामों के लिए धैर्य न रखना"
  elif lang == "es":
    title = f"{primary}: Guía Completa y Consejos Prácticos"
    meta = _trim(f"Aprenda sobre {primary} y {topic} con información clara y práctica.")
    intro_txt = f"Comprender los conceptos básicos de **{primary}** es fundamental para obtener los mejores resultados en **{topic}**."
    why_txt = f"Aplicar **{primary}** adecuadamente permite optimizar procesos y evitar errores comunes."
    b1, b2, b3 = "Conocimiento práctico de aplicación inmediata", "Estructura clara para reducir errores", "Resultados medibles a largo plazo"
    p1, p2 = "Avanzar sin comprender los conceptos fundamentales", "Esperar resultados inmediatos sin constancia"
  elif lang == "fr":
    title = f"{primary} : Guide Complet et Conseils Pratiques"
    meta = _trim(f"Découvrez tout sur {primary} et {topic} avec des conseils clairs.")
    intro_txt = f"Comprendre les bases de **{primary}** est essentiel pour réussir dans **{topic}**."
    why_txt = f"L'application efficace de **{primary}** permet de gagner du temps et de maximiser les résultats."
    b1, b2, b3 = "Connaissances pratiques applicables immédiatement", "Méthode claire et structurée", "Résultats durables à long terme"
    p1, p2 = "Se précipiter sans comprendre les bases", "Attendre des résultats immédiats sans régularité"
  elif lang == "de":
    title = f"{primary}: Vollständiger Leitfaden und Tipps"
    meta = _trim(f"Lernen Sie {primary} und {topic} mit praktischen Anleitungen kennen.")
    intro_txt = f"Das Verständnis von **{primary}** ist der Schlüssel zum Erfolg bei **{topic}**."
    why_txt = f"Die richtige Anwendung von **{primary}** spart Zeit und liefert messbare Ergebnisse."
    b1, b2, b3 = "Praktisches Wissen zur sofortigen Anwendung", "Klares Konzept zur Fehlervermeidung", "Nachhaltige Erfolge"
    p1, p2 = "Voreiliges Handeln ohne Verständnis der Grundlagen", "Erwartung sofortiger Ergebnisse ohne Geduld"
  elif lang == "gu":
    title = f"{primary}: સંપૂર્ણ માર્ગદર્શિકા અને ઉપયોગ"
    meta = _trim(f"{primary} અને {topic} વિશે સરળ અને સચોટ માહિતી મેળવો.")
    intro_txt = f"**{primary}** અને **{topic}** વિશે યોગ્ય સમજણ મેળવવી એ સફળતા માટે ખૂબ જ મહત્વપૂર્ણ છે."
    why_txt = f"**{primary}** નો યોગ્ય ઉપયોગ સમય બચાવે છે અને ઉત્તમ પરિણામો આપે છે."
    b1, b2, b3 = "તુરંત અમલમાં મૂકી શકાય તેવી વ્યવહારુ માહિતી", "ભૂલો ઘટાડવા માટે સ્પષ્ટ પદ્ધતિ", "દીર્ઘકાલીન લાભો"
    p1, p2 = "મૂળભૂત બાબતો સમજ્યા વિના ઉતાવળ કરવી", "પરિણામો માટે ધીરજ ન રાખવી"
  elif lang == "mr":
    title = f"{primary}: सविस्तर मार्गदर्शन आणि माहिती"
    meta = _trim(f"{primary} आणि {topic} बद्दल महत्त्वाची आणि उपयुक्त माहिती मिळवा.")
    intro_txt = f"**{primary}** बद्दल सविस्तर ज्ञान मिळवणे आणि **{topic}** साठी त्याचा वापर करणे अत्यंत फायदेशीर आहे."
    why_txt = f"योग्य नियोजनासह **{primary}** वापरल्यास चांगले निकाल मिळतात."
    b1, b2, b3 = "त्वरित वापरण्यायोग्य व्यावहारिक ज्ञान", "चुका टाळण्यासाठी स्पष्ट आराखडा", "दीर्घकालीन यश"
    p1, p2 = "नियोजनाशिवाय काम सुरू करणे", "सातत्य न ठेवणे"
  else:
    title = f"{primary}: {guide_h}"
    meta = _trim(f"{primary} - {topic}: {intro_h}, {benefits_h}.")
    intro_txt = f"{intro_h}: **{primary}** ({topic})."
    why_txt = f"**{primary}** - {benefits_h}."
    b1, b2, b3 = f"{guide_h}", f"{how_h}", f"{conclusion_h}"
    p1, p2 = f"{pitfalls_h} 1", f"{pitfalls_h} 2"

  body = f"""# {title}

## {intro_h}

{intro_txt}

## {how_h}

{why_txt}

## {benefits_h}

- {b1}
- {b2}
- {b3}

## {pitfalls_h}

- {p1}
- {p2}

## {conclusion_h}

**{primary}** - {meta}
"""
  return title, meta, body.strip()


def build_domain_faqs(topic: str, primary: str, domain: str, seed: int, language: str | None = None) -> list[dict[str, str]]:
  lang = detect_language(topic, [primary], language)
  if lang != "en":
    from app.engine import seo_content_engine
    return seo_content_engine.build_faqs(topic, [primary], language=lang)

  if domain == "fitness":
    variants = [
      [
        {
          "question": f"What is the best {primary} for beginners?",
          "answer": "A combination of squats, push-ups, lunges, planks, and stretching provides a balanced beginner workout.",
        },
        {
          "question": "Can beginners build muscle with home workouts?",
          "answer": "Yes. Bodyweight exercises and progressive overload help beginners build muscle and improve strength.",
        },
        {
          "question": "How long should a beginner exercise at home?",
          "answer": "Aim for 20 to 45 minutes of exercise, three to five days per week.",
        },
        {
          "question": "Do I need equipment for a home workout?",
          "answer": "No. Most beginner exercises can be performed using body weight alone.",
        },
        {
          "question": "How long does it take to see results from home workouts?",
          "answer": "Most people notice improvements in energy and fitness within four to eight weeks of consistent training.",
        },
      ],
      [
        {
          "question": f"How often should I do a {primary}?",
          "answer": "Three to five sessions per week with rest days allows muscles to recover and adapt.",
        },
        {
          "question": "Is warming up necessary for beginners?",
          "answer": "Yes. Five minutes of light movement reduces injury risk and improves performance.",
        },
        {
          "question": f"What exercises are safest for a {primary}?",
          "answer": "Bodyweight squats, incline push-ups, lunges, and planks are safe, effective starter movements.",
        },
        {
          "question": "Can I lose weight with home workouts alone?",
          "answer": "Exercise combined with a balanced diet supports weight management; consistency matters most.",
        },
      ],
    ]
    return variants[seed % len(variants)]

  if domain == "enterprise":
    return [
      {
        "question": f"What is {primary}?",
        "answer": (
          f"{primary.title()} is integrated business software that connects finance, inventory, "
          "manufacturing, HR, and reporting in one platform."
        ),
      },
      {
        "question": f"What modules are included in {primary}?",
        "answer": "Typical modules include accounting, inventory, procurement, manufacturing, CRM, and HR/payroll.",
      },
      {
        "question": f"How long does {primary} implementation take?",
        "answer": "Small businesses may go live in 3–6 months; mid-size deployments often take 6–12 months depending on scope.",
      },
      {
        "question": f"Cloud or on-premise {primary} — which is better?",
        "answer": "Cloud ERP offers faster updates and lower upfront cost; on-premise suits strict data-residency or heavy customization needs.",
      },
      {
        "question": f"Who needs {primary}?",
        "answer": "Growing companies with complex inventory, multi-location operations, or manual reconciliation pain benefit most.",
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


def build_rich_content(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  tone: str,
  audience: str | None,
  seed: int,
  language: str | None = None,
) -> dict[str, Any]:
  lang = detect_language(topic, keywords, language)
  domain = detect_domain(topic, keywords)
  kw = expand_keywords(topic, keywords, domain)
  primary = kw["primary"]
  outline = build_structured_outline(topic, primary, domain=domain, category=category, seed=seed)

  if lang != "en":
    title, meta, article = _localized_article(
      topic, primary, domain=domain, tone=tone, seed=seed, audience=audience, lang=lang, category=category,
    )
  elif domain == "fitness":
    title, meta, article = _fitness_article(topic, primary, tone, seed)
  elif domain == "enterprise":
    title, meta, article = _enterprise_article(topic, primary, tone, seed, audience)
  else:
    title, meta, article = _general_article(
      topic, primary, domain=domain, tone=tone, seed=seed, audience=audience,
    )

  faqs = build_domain_faqs(topic, primary, domain, seed, language=lang)
  return {
    "metadata": {"title": title, "meta_description": meta},
    "keywords": kw,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
    "domain": domain,
    "variation_seed": seed,
  }


def _trim(meta: str, limit: int = 160) -> str:
  meta = re.sub(r"\s+", " ", meta.strip())
  if len(meta) <= limit:
    return meta
  return meta[: limit - 3].rsplit(" ", 1)[0].rstrip() + "..."

