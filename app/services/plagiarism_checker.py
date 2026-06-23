"""Copyright & plagiarism analysis and dynamic rewrite — no GPT/Claude/Gemini."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from app.services.provider_base import ModelProvider

_STOP = frozenset(
  "a an the and or but in on at to for of is are was were be been being "
  "it this that with as by from their our your its".split()
)

_PHRASE_SPIN: list[tuple[re.Pattern[str], str]] = [
  (re.compile(r"\bartificial intelligence\b", re.I), "machine learning"),
  (re.compile(r"\btransforming\b", re.I), "reshaping"),
  (re.compile(r"\btransform\b", re.I), "reshape"),
  (re.compile(r"\btransforms\b", re.I), "reshapes"),
  (re.compile(r"\bindustry\b", re.I), "sector"),
  (re.compile(r"\bindustries\b", re.I), "sectors"),
  (re.compile(r"\bfuture\b", re.I), "outlook"),
  (re.compile(r"\bworld\b", re.I), "global landscape"),
  (re.compile(r"\bchange\b", re.I), "shift"),
  (re.compile(r"\beverything\b", re.I), "the landscape"),
  (re.compile(r"\btechnology\b", re.I), "innovation"),
  (re.compile(r"\bsolutions?\b", re.I), "approaches"),
  (re.compile(r"\bcontent\b", re.I), "material"),
  (re.compile(r"\bwebsite\b", re.I), "online platform"),
  (re.compile(r"\bbusiness\b", re.I), "enterprise"),
  (re.compile(r"\bcompany\b", re.I), "organization"),
  (re.compile(r"\bimportant\b", re.I), "essential"),
  (re.compile(r"\bhelp\b", re.I), "support"),
  (re.compile(r"\bpeople\b", re.I), "users"),
  (re.compile(r"\bwork\b", re.I), "operate"),
  (re.compile(r"\bnew\b", re.I), "recent"),
  (re.compile(r"\bgood\b", re.I), "effective"),
  (re.compile(r"\bgreat\b", re.I), "strong"),
  (re.compile(r"\buse\b", re.I), "apply"),
  (re.compile(r"\bmake\b", re.I), "build"),
  (re.compile(r"\bmany\b", re.I), "numerous"),
  (re.compile(r"\bbig\b", re.I), "major"),
  (re.compile(r"\bshow\b", re.I), "demonstrate"),
]

_CLICHÉ_PATTERNS: list[tuple[re.Pattern[str], str]] = [
  (re.compile(r"\bin today'?s digital age\b", re.I), "In the current era"),
  (re.compile(r"\bin this day and age\b", re.I), "Today"),
  (re.compile(r"\bit goes without saying\b", re.I), "Clearly"),
  (re.compile(r"\bat the end of the day\b", re.I), "Ultimately"),
  (re.compile(r"\bcutting-edge technology\b", re.I), "modern technology"),
  (re.compile(r"\bworld-class solution\b", re.I), "effective solution"),
  (re.compile(r"\bbest in class\b", re.I), "top-tier"),
  (re.compile(r"\bleverage synergies\b", re.I), "work together efficiently"),
  (re.compile(r"\bgame changer\b", re.I), "major improvement"),
  (re.compile(r"\bparadigm shift\b", re.I), "significant change"),
  (re.compile(r"\bneedless to say\b", re.I), "Naturally"),
  (re.compile(r"\bwhen it comes to\b", re.I), "Regarding"),
  (re.compile(r"\bclick here\b", re.I), "see details"),
  (re.compile(r"\bread more\b", re.I), "learn more"),
  (re.compile(r"\bas an ai language model\b", re.I), ""),
  (re.compile(r"\blorem ipsum\b", re.I), ""),
]

_SYNONYMS: dict[str, list[str]] = {
  "important": ["significant", "essential", "crucial", "vital"],
  "help": ["assist", "support", "enable", "aid"],
  "use": ["employ", "apply", "utilize", "adopt"],
  "make": ["create", "produce", "build", "develop"],
  "good": ["strong", "solid", "effective", "valuable"],
  "great": ["excellent", "outstanding", "remarkable", "superior"],
  "big": ["large", "major", "substantial", "considerable"],
  "new": ["recent", "fresh", "novel", "latest"],
  "many": ["numerous", "multiple", "several", "various"],
  "people": ["individuals", "users", "readers", "professionals"],
  "company": ["organization", "business", "firm", "enterprise"],
  "work": ["function", "operate", "perform", "run"],
  "show": ["demonstrate", "reveal", "indicate", "illustrate"],
  "change": ["transform", "shift", "alter", "evolve"],
  "future": ["ahead", "coming years", "long term", "next phase"],
  "technology": ["tech", "innovation", "digital tools", "systems"],
  "business": ["commerce", "industry", "market", "sector"],
  "content": ["material", "copy", "text", "writing"],
  "website": ["site", "web page", "online presence", "portal"],
  "service": ["offering", "solution", "package", "program"],
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _words(text: str) -> list[str]:
  return re.findall(r"\w+", text, flags=re.UNICODE)


def _sentences(text: str) -> list[str]:
  parts = _SENTENCE_SPLIT.split((text or "").strip())
  return [s.strip() for s in parts if len(s.strip()) > 12]


def _shingles(text: str, size: int = 4) -> Counter[tuple[str, ...]]:
  words = [w.lower() for w in _words(text)]
  if len(words) < size:
    return Counter()
  return Counter(tuple(words[i : i + size]) for i in range(len(words) - size + 1))


def _snippet(text: str, limit: int = 140) -> str:
  t = re.sub(r"\s+", " ", (text or "").strip())
  return t if len(t) <= limit else t[: limit - 3].rstrip() + "..."


def _jaccard(a: str, b: str) -> float:
  wa = set(w.lower() for w in _words(a) if w.lower() not in _STOP)
  wb = set(w.lower() for w in _words(b) if w.lower() not in _STOP)
  if not wa or not wb:
    return 0.0
  return len(wa & wb) / len(wa | wb)


def _text_overlap_percent(original: str, rewritten: str) -> int:
  """How much rewritten text still overlaps the original (3-word shingles)."""
  a = set(_shingles(original, 3).keys())
  b = set(_shingles(rewritten, 3).keys())
  if not a:
    return 0
  return int(round(100 * len(a & b) / len(a)))


def _pick_synonym(word: str, salt: str) -> str:
  low = word.lower()
  opts = _SYNONYMS.get(low)
  if not opts:
    return word
  idx = int(hashlib.md5(f"{salt}:{low}".encode()).hexdigest(), 16) % len(opts)
  repl = opts[idx]
  if word[0].isupper():
    return repl.capitalize()
  return repl


def _spin_sentence(sentence: str, salt: str, *, aggressive: bool = False) -> str:
  out = sentence
  for pat, repl in _PHRASE_SPIN:
    out = pat.sub(repl, out)
  for pat, repl in _CLICHÉ_PATTERNS:
    out = pat.sub(repl, out)
  out = re.sub(r"\s{2,}", " ", out).strip()

  raw_words = re.findall(r"\S+", out)
  spun_words: list[str] = []
  for i, w in enumerate(raw_words):
    core = re.sub(r"[^\w]", "", w)
    punct = w[len(core):] if core else ""
    if core and (aggressive or len(core) > 3) and core.lower() not in _STOP:
      if core.lower() in _SYNONYMS or aggressive:
        spun_words.append(_pick_synonym(core, f"{salt}:{i}") + punct)
        continue
    spun_words.append(w)
  out = " ".join(spun_words)

  if aggressive and len(spun_words) >= 5:
    h = int(hashlib.md5(f"split:{salt}".encode()).hexdigest(), 16)
    if h % 2 == 0:
      mid = len(spun_words) // 2
      out = " ".join(spun_words[mid:] + spun_words[:mid])

  return out.strip() or sentence


def _rewrite_dynamic(text: str, analysis: dict[str, Any] | None = None, *, aggressive: bool = False) -> str:
  """Fully dynamic rewrite — works on any content without fixed templates."""
  sents = _sentences(text)
  if not sents:
    return _spin_sentence(text, text[:40], aggressive=aggressive)

  seen_norm: set[str] = set()
  out_sents: list[str] = []

  flagged_snippets = {
    re.sub(r"\s+", " ", (seg.get("text") or "").lower())
    for seg in (analysis or {}).get("matched_segments", [])
  }

  for i, sent in enumerate(sents):
    norm = re.sub(r"\s+", " ", sent.lower())
    is_dup = norm in seen_norm
    is_flagged = aggressive or is_dup or any(
      fs and (fs in norm or _jaccard(fs, sent) > 0.45) for fs in flagged_snippets
    )

    if is_flagged:
      rewritten = _spin_sentence(sent, f"{text[:30]}:{i}", aggressive=True)
      norm_new = re.sub(r"\s+", " ", rewritten.lower())
      attempt = 0
      while (norm_new in seen_norm or _jaccard(rewritten, sent) > 0.52) and attempt < 8:
        rewritten = _spin_sentence(sent, f"{text[:30]}:{i}:r{attempt}", aggressive=True)
        norm_new = re.sub(r"\s+", " ", rewritten.lower())
        attempt += 1
      out_sents.append(rewritten)
      seen_norm.add(norm_new)
    else:
      out_sents.append(sent)
      seen_norm.add(norm)

  result = " ".join(out_sents)
  result = re.sub(r"\s+([.!?])", r"\1", result)
  result = re.sub(r"\s{2,}", " ", result).strip()
  return result or text


def _analyze(text: str, *, original_ref: str | None = None) -> dict[str, Any]:
  words = _words(text)
  word_count = len(words)
  unique_words = len(set(w.lower() for w in words))
  uniqueness_ratio = unique_words / max(word_count, 1)

  sents = _sentences(text)
  dup_sentences: list[str] = []
  seen: set[str] = set()
  for s in sents:
    key = re.sub(r"\s+", " ", s.lower())
    if key in seen:
      dup_sentences.append(s)
    seen.add(key)

  shingles = _shingles(text)
  total_shingles = sum(shingles.values())
  repeated_shingle_count = sum(c - 1 for c in shingles.values() if c > 1)
  repeat_ratio = repeated_shingle_count / max(total_shingles, 1)

  matched_segments: list[dict[str, Any]] = []
  low = text.lower()

  for pat, _ in _CLICHÉ_PATTERNS:
    m = pat.search(text)
    if m:
      matched_segments.append({
        "type": "similar_content",
        "label": "Known web phrase",
        "text": _snippet(m.group(0)),
        "match_percent": 92,
        "source": "Commonly published online wording",
      })

  for i, a in enumerate(sents):
    for b in sents[i + 1 :]:
      sim = _jaccard(a, b)
      if sim >= 0.72 and len(a) > 25:
        matched_segments.append({
          "type": "near_duplicate",
          "label": "Highly similar sentences",
          "text": _snippet(a),
          "match_percent": int(sim * 100),
          "source": "Overlaps another sentence in your text",
        })
        break

  for s in dup_sentences[:8]:
    matched_segments.append({
      "type": "duplicate",
      "label": "Duplicate sentence",
      "text": _snippet(s),
      "match_percent": 100,
      "source": "Exact repeat within your text",
    })

  for gram, count in shingles.most_common(8):
    if count < 2:
      break
    phrase = " ".join(gram)
    if len(phrase) < 10:
      continue
    matched_segments.append({
      "type": "repeated_sequence",
      "label": "Repeated phrase",
      "text": phrase,
      "match_percent": min(96, 55 + count * 10),
      "source": f"Repeated {count} times",
    })

  dup_ratio = len(dup_sentences) / max(len(sents), 1)
  phrase_hits = len([seg for seg in matched_segments if seg["type"] == "similar_content"])
  phrase_ratio = min(1.0, phrase_hits * 0.18)
  near_dup_ratio = min(1.0, len([s for s in matched_segments if s["type"] == "near_duplicate"]) * 0.2)
  vocab_gap = max(0.0, 0.55 - uniqueness_ratio) / 0.55

  similarity_percent = int(round(min(
    98,
    max(0, dup_ratio * 32 + phrase_ratio * 28 + near_dup_ratio * 22 + repeat_ratio * 18 + vocab_gap * 12),
  )))

  if original_ref:
    overlap = _text_overlap_percent(original_ref, text)
    similarity_percent = max(similarity_percent, min(90, int(overlap * 0.85)))

  original_percent = 100 - similarity_percent
  risk = "low" if similarity_percent < 20 else "medium" if similarity_percent < 50 else "high"

  flags: list[str] = []
  if similarity_percent >= 15:
    flags.append(f"{similarity_percent}% similarity with known or repeated patterns")
  if dup_sentences:
    flags.append(f"{len(dup_sentences)} duplicate sentence(s)")
  if phrase_hits:
    flags.append(f"{phrase_hits} common web phrase(s) detected")
  if original_ref and _text_overlap_percent(original_ref, text) > 35:
    flags.append("High overlap with original draft — rewrite further")

  suggestions: list[str] = []
  if similarity_percent >= 12:
    suggestions.append("Click Remove Plagiarism for a fully rewritten version.")
  if not suggestions:
    suggestions.append("Content passes heuristic originality checks. Verify manually before publishing.")

  return {
    "word_count": word_count,
    "unique_words": unique_words,
    "sentence_count": len(sents),
    "similarity_percent": similarity_percent,
    "original_percent": original_percent,
    "originality_score": original_percent,
    "risk_level": risk,
    "likely_original": similarity_percent < 12,
    "content_preview": _snippet(text, 48),
    "matched_segments": matched_segments[:15],
    "duplicate_sentences": [_snippet(s) for s in dup_sentences[:6]],
    "repeated_sequences": [" ".join(g) for g, c in shingles.most_common(6) if c > 1],
    "flags": flags,
    "suggestions": suggestions,
    "summary": f"{similarity_percent}% similarity detected · {original_percent}% original content",
    "overlap_with_original_percent": _text_overlap_percent(original_ref, text) if original_ref else 0,
  }


async def check_content(*, content: str, settings: Any | None = None) -> dict[str, Any]:
  text = (content or "").strip()
  if len(text) < 40:
    raise ValueError("content must be at least 40 characters")
  return await _analyze_full(text, settings=settings)


async def _analyze_full(
  text: str,
  *,
  original_ref: str | None = None,
  settings: Any | None = None,
) -> dict[str, Any]:
  result = _analyze(text, original_ref=original_ref)

  try:
    from app.config import get_settings
    from app.engine.plagiarism_engine import configure, is_available, scan_content

    cfg = settings or get_settings()
    configure(cfg.plagiarism_index_dir)

    emb = await scan_content(
      text,
      searxng_url=cfg.searxng_url if cfg.plagiarism_use_searxng else "",
      use_live_wikipedia=cfg.plagiarism_use_wikipedia_live,
      use_searxng=cfg.plagiarism_use_searxng,
    )

    if emb.get("available"):
      result["embedding_available"] = True
      result["sources_used"] = emb.get("sources_used") or []
      result["avg_embedding_similarity"] = emb.get("avg_embedding_similarity", 0)
      result["chunks_scanned"] = emb.get("chunks_scanned", 0)
      result["chunks_matched"] = emb.get("chunks_matched", 0)

      emb_sim = int(emb.get("similarity_percent") or 0)
      result["similarity_percent"] = max(result["similarity_percent"], emb_sim)
      result["original_percent"] = 100 - result["similarity_percent"]
      result["originality_score"] = result["original_percent"]
      result["likely_original"] = result["similarity_percent"] < 12

      emb_segments = emb.get("matched_segments") or []
      emb_highlights = emb.get("highlighted_sentences") or []
      result["workflow"] = emb.get("workflow") or []
      result["highlighted_sentences"] = emb_highlights
      result["scan_incomplete"] = bool(emb.get("scan_incomplete"))
      sr = emb.get("similarity_report") or {}
      result["similarity_report"] = sr if sr else None

      merged = emb_segments + result.get("matched_segments", [])
      seen: set[str] = set()
      deduped: list[dict[str, Any]] = []
      for seg in merged:
        key = (seg.get("text") or "")[:60]
        if key in seen:
          continue
        seen.add(key)
        deduped.append(seg)
      result["matched_segments"] = deduped[:20]

      if emb_sim >= 15:
        result["flags"] = list(result.get("flags") or [])
        if not any("embedding" in f.lower() or "wikipedia" in f.lower() or "web" in f.lower() for f in result["flags"]):
          result["flags"].append(
            f"Semantic match: {emb.get('chunks_matched', 0)} chunk(s) similar to indexed/web content"
          )
    else:
      result["embedding_available"] = False
      result["sources_used"] = ["heuristic"]
      if emb.get("error"):
        result["embedding_note"] = emb["error"]
      result["scan_incomplete"] = True
      result["flags"] = list(result.get("flags") or [])
      result["flags"].append(
        "Semantic scan unavailable — install sentence-transformers + faiss-cpu, "
        "then run: python scripts/build_plagiarism_index.py"
      )
    if emb.get("available") and emb.get("scan_incomplete"):
      result["scan_incomplete"] = True
      result["flags"] = list(result.get("flags") or [])
      if not any("reference sources" in f.lower() for f in result["flags"]):
        result["flags"].append(
          "No reference sources reached (Wikipedia/FAISS). Similarity may be understated. "
          "Check internet and run: python scripts/build_plagiarism_index.py"
        )
    if not is_available() and result.get("embedding_available") is not True:
      result.setdefault("embedding_note", "Run: python scripts/build_plagiarism_index.py")
  except ImportError:
    result["embedding_available"] = False
    result["sources_used"] = ["heuristic"]
    result["embedding_note"] = "pip install sentence-transformers faiss-cpu"

  result["summary"] = (
    f"{result['similarity_percent']}% similarity detected · "
    f"{result['original_percent']}% original content"
  )
  return result


_REWRITE_SYSTEM = (
  "Rewrite the text in completely fresh wording. Same meaning, new sentences. "
  "No clichés. Output only the rewritten body."
)


async def _try_model_rewrite(provider: ModelProvider, text: str, issues: list[str]) -> str | None:
  try:
    issue_block = "\n".join(f"- {i}" for i in issues) or "- Improve originality"
    raw = await provider.chat(
      [{"role": "user", "content": f"Issues:\n{issue_block}\n\nRewrite:\n{text}"}],
      system_prompt=_REWRITE_SYSTEM,
      use_rag=False,
      skip_intent=True,
      max_tokens=min(1600, max(350, len(text) // 2 + 150)),
      temperature=0.55,
    )
    cleaned = re.sub(r"\s+", " ", (raw or "").strip())
    if len(cleaned) >= 30 and not cleaned.lower().startswith("i could not"):
      return cleaned
  except Exception:
    pass
  return None


async def remove_plagiarism(
  provider: ModelProvider,
  *,
  content: str,
) -> dict[str, Any]:
  text = (content or "").strip()
  if len(text) < 40:
    raise ValueError("content must be at least 40 characters")

  before = await _analyze_full(text)
  rewritten = _rewrite_dynamic(text, before, aggressive=True)

  for pass_n in range(6):
    after_probe = await _analyze_full(rewritten, original_ref=text)
    if after_probe["similarity_percent"] <= 12 and after_probe["overlap_with_original_percent"] <= 28:
      break
    rewritten = _rewrite_dynamic(rewritten, after_probe, aggressive=True)

  model_text = await _try_model_rewrite(provider, rewritten, before.get("flags", []))
  if model_text:
    model_after = await _analyze_full(model_text, original_ref=text)
    algo_after = await _analyze_full(rewritten, original_ref=text)
    if model_after["similarity_percent"] <= algo_after["similarity_percent"]:
      rewritten = model_text
    elif model_after["overlap_with_original_percent"] < algo_after["overlap_with_original_percent"]:
      rewritten = model_text

  for _ in range(3):
    after_probe = await _analyze_full(rewritten, original_ref=text)
    if after_probe["similarity_percent"] <= 12:
      break
    rewritten = _rewrite_dynamic(rewritten, after_probe, aggressive=True)

  after = await _analyze_full(rewritten, original_ref=text)
  return {
    "original_content": text,
    "rewritten_content": rewritten,
    "before": before,
    "after": after,
    "similarity_percent_before": before["similarity_percent"],
    "similarity_percent_after": after["similarity_percent"],
    "original_percent_before": before["original_percent"],
    "original_percent_after": after["original_percent"],
    "overlap_with_original_percent": after["overlap_with_original_percent"],
    "improvement": before["similarity_percent"] - after["similarity_percent"],
    "summary": (
      f"Similarity {before['similarity_percent']}% → {after['similarity_percent']}% · "
      f"{after['original_percent']}% original · "
      f"{after['overlap_with_original_percent']}% overlap with your original draft."
    ),
  }
