"""SEO Optimizer engine — analysis metrics, categories, multilingual training.

100% custom — no GPT, Claude, Gemini.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.engine.knowledge import KnowledgeBase, load_knowledge_base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEO_OPTIMIZER_KB_PATH = PROJECT_ROOT / "data" / "seo_optimizer_knowledge.jsonl"

_VALID_TONES = ["professional", "casual", "friendly", "formal"]

_TONE_HINTS: dict[str, str] = {
  "professional": "Clear, confident, business-appropriate.",
  "casual": "Relaxed, conversational, approachable.",
  "friendly": "Warm, helpful, welcoming.",
  "formal": "Structured, respectful, corporate or academic.",
}

_CATEGORIES: dict[str, dict[str, str]] = {
  "blog_article": {"label": "Blog Article", "default_tone": "professional"},
  "landing_page": {"label": "Landing Page", "default_tone": "professional"},
  "product_description": {"label": "Product Description", "default_tone": "professional"},
  "email_copy": {"label": "Email Copy", "default_tone": "friendly"},
  "social_post": {"label": "Social Post", "default_tone": "casual"},
  "local_seo": {"label": "Local SEO Page", "default_tone": "friendly"},
  "technical_doc": {"label": "Technical Documentation", "default_tone": "formal"},
  "ecommerce": {"label": "E-commerce Copy", "default_tone": "professional"},
}

_LANG_TO_BCP47: dict[str, str] = {
  "english": "en", "en": "en", "hindi": "hi", "hi": "hi",
  "spanish": "es", "es": "es", "french": "fr", "fr": "fr",
  "german": "de", "de": "de", "portuguese": "pt", "pt": "pt",
  "arabic": "ar", "ar": "ar", "japanese": "ja", "ja": "ja",
  "chinese": "zh", "zh": "zh",
}

_seo_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
  global _seo_kb
  if _seo_kb is None:
    _seo_kb = load_knowledge_base(knowledge_path=SEO_OPTIMIZER_KB_PATH)
  return _seo_kb


def reload_kb() -> KnowledgeBase:
  """Reload training data after import (no server restart needed in dev)."""
  global _seo_kb
  _seo_kb = load_knowledge_base(knowledge_path=SEO_OPTIMIZER_KB_PATH)
  return _seo_kb


def bcp47(language: str | None) -> str:
  if not language:
    return "en"
  return _LANG_TO_BCP47.get(language.strip().lower(), language.strip().lower()[:5] or "en")


def normalize_tone(tone: str | None, category: str | None = None) -> str:
  if tone and tone.strip().lower() in _VALID_TONES:
    return tone.strip().lower()
  cat = normalize_category(category)
  default = _CATEGORIES.get(cat, {}).get("default_tone", "professional")
  return default if default in _VALID_TONES else "professional"


def normalize_category(category: str | None) -> str:
  if not category:
    return "blog_article"
  key = category.strip().lower().replace(" ", "_").replace("-", "_")
  return key if key in _CATEGORIES else "blog_article"


def tone_hint(tone: str) -> str:
  return _TONE_HINTS.get(tone, _TONE_HINTS["professional"])


def supported_categories() -> list[dict[str, str]]:
  return [{"id": k, **v} for k, v in _CATEGORIES.items()]


def supported_tones() -> list[dict[str, str]]:
  return [{"id": t, "label": t.capitalize()} for t in _VALID_TONES]


def supported_languages() -> list[dict[str, str]]:
  return [
    {"name": "English", "code": "en"}, {"name": "Hindi", "code": "hi"},
    {"name": "Spanish", "code": "es"}, {"name": "French", "code": "fr"},
    {"name": "German", "code": "de"}, {"name": "Portuguese", "code": "pt"},
    {"name": "Arabic", "code": "ar"}, {"name": "Japanese", "code": "ja"},
    {"name": "Chinese", "code": "zh"},
  ]


def _syllables(word: str) -> int:
  w = word.lower().strip(".,!?;:'\"")
  if len(w) <= 2:
    return 1
  vowels = "aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿ"
  count = 0
  prev_v = False
  for ch in w:
    is_v = ch in vowels
    if is_v and not prev_v:
      count += 1
    prev_v = is_v
  return max(1, count)


def count_sentences(text: str) -> int:
  parts = re.split(r"[.!?]+", text or "")
  return max(1, len([p for p in parts if p.strip()]))


def count_words(text: str) -> int:
  return len(re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE))


def count_characters(text: str) -> int:
  return len((text or "").strip())


def _prose_only(text: str) -> str:
  chunks: list[str] = []
  for line in (text or "").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    line = re.sub(r"^[\*\-]\s+", "", line)
    chunks.append(line)
  return " ".join(chunks)


def _flesch_raw(prose: str) -> float:
  words = re.findall(r"\b[\w'-]+\b", prose or "", flags=re.UNICODE)
  if not words:
    return 30.0
  sentences = max(1, len([p for p in re.split(r"[.!?]+", prose) if p.strip()]))
  syllable_count = sum(_syllables(w) for w in words)
  asl = len(words) / sentences
  asw = syllable_count / len(words)
  return 206.835 - (1.015 * asl) - (84.6 * asw)


def readability_score(text: str) -> float:
  """SEO content readability index (0–100; target 60–95 for optimized articles)."""
  raw = text or ""
  prose = _prose_only(raw)
  if not count_words(prose):
    return 50.0

  flesch = _flesch_raw(prose)
  # Technical web copy often scores 5–35 on raw Flesch — rescale to a usable band.
  base = 38.0 + max(0.0, min(42.0, flesch * 1.15))

  bonus = 0.0
  if re.search(r"^##\s+", raw, re.MULTILINE):
    bonus += 10.0
  if re.search(r"^###\s+", raw, re.MULTILINE):
    bonus += 4.0
  bullets = len(re.findall(r"^[\*\-]\s+", raw, re.MULTILINE))
  bonus += min(8.0, bullets * 1.5)

  sents = [s.strip() for s in re.split(r"[.!?]+", prose) if s.strip()]
  if sents:
    avg_sent = sum(count_words(s) for s in sents) / len(sents)
    if avg_sent <= 16:
      bonus += 14.0
    elif avg_sent <= 20:
      bonus += 11.0
    elif avg_sent <= 24:
      bonus += 7.0
    elif avg_sent <= 28:
      bonus += 3.0
    else:
      bonus -= 6.0

  paras = [
    p for p in re.split(r"\n\s*\n", raw)
    if p.strip() and not p.strip().startswith("#") and count_words(p) > 0
  ]
  if paras:
    avg_para = sum(count_words(p) for p in paras) / len(paras)
    if avg_para <= 60:
      bonus += 10.0
    elif avg_para <= 85:
      bonus += 6.0
    elif avg_para <= 110:
      bonus += 2.0
    elif avg_para > 130:
      bonus -= 5.0

  return round(max(0.0, min(100.0, base + bonus)), 2)


def _split_long_sentence(sentence: str, max_words: int = 20) -> str:
  sentence = sentence.strip()
  if count_words(sentence) <= max_words:
    return sentence
  for delim in (", ", "; ", " — ", " - ", " and ", " while ", " which ", " that ", " because "):
    if delim not in sentence:
      continue
    left, right = sentence.split(delim, 1)
    if count_words(left) >= 7 and count_words(right) >= 5:
      end = "." if not left.rstrip().endswith((".", "!", "?")) else ""
      return f"{left.rstrip('.,')}{end} {right.lstrip()}"
  words = sentence.split()
  mid = len(words) // 2
  return f"{' '.join(words[:mid]).rstrip('.,')}. {' '.join(words[mid:])}"


def _shorten_prose_block(block: str, max_sent_words: int = 20) -> str:
  if block.strip().startswith("#"):
    return block
  lines: list[str] = []
  for line in block.splitlines():
    stripped = line.strip()
    if not stripped:
      lines.append(line)
      continue
    if stripped.startswith("#"):
      lines.append(line)
      continue
    prefix = ""
    body = stripped
    if re.match(r"^[\*\-]\s+", stripped):
      prefix = re.match(r"^([\*\-]\s+)", stripped).group(1)  # type: ignore[union-attr]
      body = stripped[len(prefix) :]
    sents = re.split(r"(?<=[.!?])\s+", body)
    fixed = [_split_long_sentence(s, max_sent_words) for s in sents if s.strip()]
    lines.append(prefix + " ".join(fixed))
  return "\n".join(lines)


def improve_readability(
  text: str,
  *,
  target_min: float = 60.0,
  target_max: float = 95.0,
  max_passes: int = 2,
) -> tuple[str, list[str]]:
  """Light polish for shorter sentences/paragraphs (non-blocking; max 2 passes)."""
  suggestions: list[str] = []
  text = re.sub(r"\b(very|really|just|actually|basically)\b\s+", "", text or "", flags=re.I)

  for _ in range(max(1, max_passes)):
    score = readability_score(text)
    if score >= target_min:
      break

    paras = re.split(r"\n\s*\n", text)
    changed = False
    fixed: list[str] = []
    for para in paras:
      if para.strip().startswith("#"):
        fixed.append(para)
        continue
      if count_words(para) > 85:
        sents = re.split(r"(?<=[.!?])\s+", para)
        mid = max(1, len(sents) // 2)
        fixed.append(" ".join(sents[:mid]))
        fixed.append(" ".join(sents[mid:]))
        changed = True
        suggestions.append("Split an oversized paragraph for readability.")
      else:
        shortened = _shorten_prose_block(para)
        if shortened != para:
          changed = True
          suggestions.append("Shortened long sentence(s) to improve flow.")
        fixed.append(shortened)

    text = "\n\n".join(fixed)
    if not changed:
      break

  score = readability_score(text)
  if score < target_min:
    suggestions.append(f"Readability {score}/100 — consider shorter sentences or more bullets (target {target_min:.0f}+).")
  elif score > target_max:
    suggestions.append(f"Readability is high ({score}/100).")

  text = re.sub(r"\n{3,}", "\n\n", text).strip()
  return text, suggestions


def content_metrics(text: str) -> dict[str, Any]:
  return {
    "readability_score": readability_score(text),
    "word_count": count_words(text),
    "character_count": count_characters(text),
    "sentence_count": count_sentences(text),
  }


def get_lsi_synonyms(keyword: str, language: str = "en") -> list[str]:
  """Derive LSI terms, stem variations, and semantic synonyms for keyword coverage analysis."""
  kw = (keyword or "").strip().lower()
  if not kw:
    return []

  synonyms: set[str] = set()
  # Standard stem / plural variations
  if kw.endswith("s") and len(kw) > 3:
    synonyms.add(kw[:-1])
  elif len(kw) > 3:
    synonyms.add(kw + "s")

  if kw.endswith("ing") and len(kw) > 5:
    synonyms.add(kw[:-3])

  # Common LSI pairings by keyword patterns
  lsi_map = {
    "mobile repair": ["smartphone repair", "cell phone fix", "mobile servicing", "phone maintenance"],
    "seo": ["search engine optimization", "organic ranking", "search traffic", "keyword strategy"],
    "python": ["python programming", "python language", "coding in python", "python script"],
    "coffee": ["coffee brewing", "espresso", "roasted coffee", "coffee maker"],
    "મોબાઈલ સંભાળ": ["સ્માર્ટફોન સંભાળ", "મોબાઈલ રીપેરિંગ", "ફોન જાળવણી", "મોબાઈલ ટીપ્સ"],
    "कॉफी कैसे बनाएं": ["कॉफी बनाने का तरीका", "घर पर कॉफी", "कॉफी रेसिपी", "स्पेशल कॉफी"],
  }

  for k, v in lsi_map.items():
    if k in kw or kw in k:
      synonyms.update(v)

  # Extract significant N-grams
  words = re.findall(r"\b[\w'-]+\b", kw, flags=re.UNICODE)
  if len(words) > 1:
    for w in words:
      if len(w) > 3 and w not in ("how", "what", "with", "from", "your", "best", "the"):
        synonyms.add(w)

  return [s for s in synonyms if s != kw]


def analyze_issues(content: str, keywords: list[str] | None = None, language: str | None = None) -> list[dict[str, str]]:
  from app.engine.seo_content_domains import detect_language
  lang = detect_language(content, keywords or [], language)
  issues: list[dict[str, str]] = []
  text = content or ""
  wc = count_words(text)

  # Multilingual message catalogs
  msg = {
    "hi": {
      "too_short": "SEO रैंकिंग के लिए सामग्री बहुत छोटी है (कम से कम 300+ शब्द लिखें)।",
      "no_h2": "पठनीयता और रैंकिंग सुधारने के लिए H2 (##) उप-शीर्षक जोड़ें।",
      "no_h1": "शीर्ष पर एक स्पष्ट H1 शीर्षक जोड़ने पर विचार करें।",
      "long_paras": "पैराग्राफ बहुत लंबे हैं — छोटे ब्लॉकों में विभाजित करें।",
      "long_sents": "कुछ वाक्य बहुत लंबे हैं — प्रति वाक्य 15-20 शब्द रखें।",
      "primary_first_para": "मुख्य कीवर्ड '{kw}' पहले पैराग्राफ में आना चाहिए।",
      "density_high": "कीवर्ड डेंसिटी बहुत अधिक हो सकती है — स्वाभाविक प्रवाह के लिए कम करें।",
      "density_low": "मुख्य कीवर्ड '{kw}' का स्वाभाविक रूप से कुछ और बार उपयोग करें।",
      "no_conclusion": "स्पष्ट निष्कर्ष या कॉल-टू-एक्शन (CTA) के साथ एक निष्कर्ष अनुभाग जोड़ें。",
    },
    "gu": {
      "too_short": "SEO રેન્કિંગ માટે સામગ્રી ખૂબ ટૂંકી છે (ઓછામાં ઓછા 300+ શબ્દો હોવા જોઈએ).",
      "no_h2": "વાંચનક્ષમતા અને રેન્કિંગ વધારવા માટે H2 (##) પેટા-શીર્ષકો ઉમેરો.",
      "no_h1": "ટોચ પર સ્પષ્ટ H1 શીર્ષક ઉમેરવાનું વિચારો.",
      "long_paras": "ફકરા ખૂબ લાંબા છે — નાના વિભાગોમાં વિભાજીત કરો.",
      "long_sents": "કેટલાક વાક્યો ખૂબ લાંબા છે — વાક્ય દીઠ 15–20 શબ્દો રાખો.",
      "primary_first_para": "મુખ્ય કીવર્ડ '{kw}' પ્રથમ ફકરામાં આવવો જોઈએ.",
      "density_high": "કીવર્ડ ઘનતા ખૂબ વધારે હોઈ શકે છે — કુદરતી પ્રવાહ માટે ઘટાડો.",
      "density_low": "મુખ્ય કીવર્ડ '{kw}' નો કુદરતી રીતે વધુ ઉપયોગ કરો.",
      "no_conclusion": "સ્પષ્ટ સારાંશ અથવા આહ્વાન (CTA) સાથે નિષ્કર્ષ વિભાગ ઉમેરો.",
    },
    "mr": {
      "too_short": "SEO रँकिंगसाठी मजकूर खूप लहान आहे (किमान 300+ शब्द असावेत).",
      "no_h2": "वाचनीयता आणि रँकिंग सुधारण्यासाठी H2 (##) उपशीर्षके जोडा.",
      "no_h1": "वर स्पष्ट H1 शीर्षक जोडण्याचा विचार करा.",
      "long_paras": "परिच्छेद खूप मोठे आहेत — लहान भागात विभाजित करा.",
      "long_sents": "काही वाक्ये खूप लांब आहेत — दर वाक्यात 15-20 शब्द ठेवा.",
      "primary_first_para": "मुख्य कीवर्ड '{kw}' पहिल्या परिच्छेदात आला पाहिजे.",
      "density_high": "कीवर्ड प्रमाण खूप जास्त असू शकते — नैसर्गिक प्रवाहासाठी कमी करा.",
      "density_low": "मुख्य कीवर्ड '{kw}' चा नैसर्गिकपणे वापर वाढवा.",
      "no_conclusion": "स्पष्ट निष्कर्षासह एक शेवटचा भाग जोडा.",
    },
    "es": {
      "too_short": "El contenido es demasiado corto para un buen SEO (apunte a 300+ palabras).",
      "no_h2": "Añada subtítulos H2 (##) para mejorar la lectura y el posicionamiento.",
      "no_h1": "Considere agregar un título H1 claro en la parte superior.",
      "long_paras": "Párrafos demasiado largos: divídalos en bloques más cortos.",
      "long_sents": "Algunas oraciones son demasiado largas: procure 15–20 palabras por oración.",
      "primary_first_para": "La palabra clave principal '{kw}' debe aparecer en el primer párrafo.",
      "density_high": "La densidad de palabras clave es demasiado alta; redúzcala para un flujo natural.",
      "density_low": "Use la palabra clave principal '{kw}' algunas veces más de forma natural.",
      "no_conclusion": "Agregue una sección de conclusión con una llamada a la acción clara.",
    },
    "fr": {
      "too_short": "Le contenu est trop court pour un bon référencement (visez 300+ mots).",
      "no_h2": "Ajoutez des sous-titres H2 (##) pour améliorer la lisibilité et le classement.",
      "no_h1": "Pensez à ajouter un titre H1 clair en haut.",
      "long_paras": "Des paragraphes sont très me longs — divisez-les en blocs plus courts.",
      "long_sents": "Certaines phrases sont trop longues — visez 15 à 20 mots par phrase.",
      "primary_first_para": "Le mot-clé principal '{kw}' doit apparaître dans le premier paragraphe.",
      "density_high": "La densité de mots-clés est trop élevée — réduisez pour un flux naturel.",
      "density_low": "Utilisez le mot-clé principal '{kw}' quelques fois de plus naturellement.",
      "no_conclusion": "Ajoutez une section de conclusion avec un appel à l'action clair.",
    },
    "de": {
      "too_short": "Der Inhalt ist zu kurz für gutes SEO (mindestens 300+ Wörter anstreben).",
      "no_h2": "Fügen Sie H2-Unterüberschriften (##) hinzu, um die Lesbarkeit zu verbessern.",
      "no_h1": "Erwägen Sie, oben eine klare H1-Überschrift hinzuzufügen.",
      "long_paras": "Einige Absätze sind zu lang — in kürzere Blöcke aufteilen.",
      "long_sents": "Einige Sätze sind zu lang — zielen Sie auf 15–20 Wörter pro Satz ab.",
      "primary_first_para": "Das Hauptschlüsselwort '{kw}' sollte im ersten Absatz erscheinen.",
      "density_high": "Die Keyword-Dichte ist möglicherweise zu hoch — reduzieren Sie für natürlichen Fluss.",
      "density_low": "Verwenden Sie das Hauptschlüsselwort '{kw}' natürlich noch ein paar Mal.",
      "no_conclusion": "Fügen Sie einen Fazit-Abschnitt mit einem klaren CTA hinzu.",
    },
    "en": {
      "too_short": "Content is too short for strong SEO (aim for 300+ words for articles).",
      "no_h2": "Add H2 (##) subheadings to improve scanability and rankings.",
      "no_h1": "Consider adding a clear H1 title at the top.",
      "long_paras": "Paragraphs are very long — split into shorter blocks.",
      "long_sents": "Some sentences are too long — aim for 15–20 words per sentence.",
      "primary_first_para": "Primary keyword '{kw}' should appear in the first paragraph.",
      "density_high": "Keyword density may be too high — reduce stuffing for natural flow.",
      "density_low": "Use primary keyword '{kw}' a few more times naturally.",
      "no_conclusion": "Add a conclusion section with a clear takeaway or CTA.",
    },
  }

  catalog = msg.get(lang, msg["en"])

  if wc < 50:
    issues.append({"type": "length", "priority": "high", "message": catalog["too_short"]})
  if "##" not in text and wc > 150:
    issues.append({"type": "structure", "priority": "high", "message": catalog["no_h2"]})
  if not re.search(r"^#\s+", text, re.MULTILINE) and wc > 100:
    issues.append({"type": "structure", "priority": "medium", "message": catalog["no_h1"]})

  long_paras = [p for p in re.split(r"\n\s*\n", text) if count_words(p) > 120 and not p.strip().startswith("#")]
  if long_paras:
    issues.append({"type": "readability", "priority": "medium", "message": catalog["long_paras"]})

  long_sents = [s for s in re.split(r"[.!?]+", text) if count_words(s) > 30]
  if len(long_sents) >= 2:
    issues.append({"type": "readability", "priority": "medium", "message": catalog["long_sents"]})

  if keywords:
    primary = keywords[0].lower()
    if primary not in text.lower()[:400]:
      issues.append({"type": "keyword", "priority": "high", "message": catalog["primary_first_para"].format(kw=keywords[0])})
    density = text.lower().count(primary) / max(wc, 1) * 100
    if density > 3.5:
      issues.append({"type": "keyword", "priority": "high", "message": catalog["density_high"]})
    elif density < 0.3 and wc > 100:
      issues.append({"type": "keyword", "priority": "medium", "message": catalog["density_low"].format(kw=keywords[0])})

  if not re.search(r"(conclusion|summary|in summary|to sum up|finally|નિષ્કર્ષ|निष्कर्ष|resumen|fazit)", text, re.IGNORECASE) and wc > 250:
    issues.append({"type": "structure", "priority": "low", "message": catalog["no_conclusion"]})

  return issues


def seo_score_from_analysis(metrics: dict[str, Any], issues: list[dict[str, str]]) -> int:
  score = 100
  for issue in issues:
    p = issue.get("priority", "low")
    score -= {"high": 18, "medium": 10, "low": 5}.get(p, 5)
  if metrics.get("readability_score", 0) < 55:
    score -= 15
  elif metrics.get("readability_score", 0) < 65:
    score -= 8
  if metrics.get("word_count", 0) < 100:
    score -= 10
  return max(0, min(100, score))


def analyze_keyword_density(text: str, keywords: list[str], language: str | None = None) -> dict[str, Any]:
  """Analyze primary & secondary keyword density, semantic LSI synonym coverage & structural distribution."""
  raw = text or ""
  low = raw.lower()
  wc = count_words(raw)
  if not wc or not keywords:
    return {"primary_density": 0.0, "status": "unknown", "distribution": {}, "keyword_metrics": [], "lsi_keywords_detected": []}

  # Split sections for structural distribution check
  paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
  intro = paras[0].lower() if paras else ""
  headings = " ".join([ln.lower() for ln in raw.splitlines() if ln.strip().startswith("#")])
  conclusion = paras[-1].lower() if len(paras) > 1 else ""

  primary = keywords[0].strip()
  metrics_list: list[dict[str, Any]] = []
  all_lsi_terms: list[str] = []

  for kw in keywords[:6]:
    k_low = kw.strip().lower()
    if not k_low:
      continue
    kw_words = len(re.findall(r"\b[\w'-]+\b", k_low))
    occurrences = len(re.findall(r"\b" + re.escape(k_low) + r"\b", low))
    
    # Semantic LSI synonym matching
    lsi_syns = get_lsi_synonyms(kw, language or "en")
    lsi_found: list[str] = []
    lsi_occurrences = 0
    for s in lsi_syns:
      s_cnt = len(re.findall(r"\b" + re.escape(s.lower()) + r"\b", low))
      if s_cnt > 0:
        lsi_found.append(s)
        lsi_occurrences += s_cnt
        if s not in all_lsi_terms:
          all_lsi_terms.append(s)

    total_semantic = occurrences + lsi_occurrences
    density = round((occurrences * kw_words / wc) * 100, 2)
    semantic_density = round((total_semantic * kw_words / wc) * 100, 2)

    in_intro = bool(re.search(r"\b" + re.escape(k_low) + r"\b", intro))
    in_headings = bool(re.search(r"\b" + re.escape(k_low) + r"\b", headings))
    in_conclusion = bool(re.search(r"\b" + re.escape(k_low) + r"\b", conclusion))

    metrics_list.append({
      "keyword": kw,
      "count": occurrences,
      "density_percent": density,
      "semantic_count": total_semantic,
      "semantic_density_percent": semantic_density,
      "lsi_synonyms_found": lsi_found,
      "in_intro": in_intro,
      "in_headings": in_headings,
      "in_conclusion": in_conclusion,
    })

  primary_metric = metrics_list[0] if metrics_list else {"density_percent": 0.0, "semantic_density_percent": 0.0}
  pd = primary_metric.get("semantic_density_percent") or primary_metric.get("density_percent", 0.0)

  if pd < 0.5:
    status = "underused"
  elif 0.5 <= pd <= 2.5:
    status = "optimal"
  else:
    status = "overused"

  return {
    "primary_keyword": primary,
    "primary_density": pd,
    "status": status,
    "optimal_range": "1.0% - 1.8%",
    "keyword_metrics": metrics_list,
    "lsi_keywords_detected": all_lsi_terms,
  }


