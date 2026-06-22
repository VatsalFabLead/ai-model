"""AI Email Assistant helpers.

Supports:
- New email generation
- Reply generation
- Cold email generation

Uses your active custom provider backend and robust parsing/cleanup for small
models. No GPT/Claude/Gemini involved.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.services.provider_base import ModelProvider

_VALID_TONES = {
  "professional",
  "casual",
  "friendly",
  "formal",
  "empathetic",
  "confident",
  "persuasive",
  "neutral",
}


def _normalize_tone(tone: str | None) -> str:
  if not tone:
    return "professional"
  t = tone.strip().lower()
  return t if t in _VALID_TONES else "professional"


def _unwrap_json_like(text: str) -> str:
  t = (text or "").strip()
  if t.startswith("```"):
    t = re.sub(r"^```(?:json|markdown|md)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
  if t.startswith("{"):
    try:
      obj = json.loads(t)
      if isinstance(obj, dict):
        for key in ("email", "body", "content", "message", "text"):
          val = obj.get(key)
          if isinstance(val, str) and val.strip():
            return val.strip()
    except Exception:
      pass
  return t


def _clean_email(text: str) -> str:
  text = _unwrap_json_like(text)
  text = text.strip()
  text = re.sub(
    r"^(sure[,!.]?\s+)?(here(?:'s| is)|certainly|of course)[^\n:]*:\s*",
    "",
    text,
    flags=re.IGNORECASE,
  )
  text = re.sub(r"^\s*(subject(?:\s*line)?|email|body|reply)\s*:\s*", "", text, flags=re.IGNORECASE)
  # If model emits a leading "Subject Line:" block, strip first line pair.
  text = re.sub(r"^\s*subject(?:\s*line)?\s*[:\-].*\n+", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\n{3,}", "\n\n", text).strip()
  return text


def _subject_from_body(body: str, fallback: str = "Re: Update") -> str:
  lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
  if not lines:
    return fallback
  first = re.sub(r"^#+\s*", "", lines[0]).strip(" -*_`:.")
  if len(first) > 70:
    first = first[:67].rsplit(" ", 1)[0] + "..."
  return first if first else fallback


async def generate_new_email(
  provider: ModelProvider,
  *,
  subject: str,
  context: str,
  tone: str | None = None,
  language: str | None = None,
) -> dict[str, Any]:
  tone_str = _normalize_tone(tone)
  lang_line = f"Write in {language}." if language else "Write in clear business English."
  subject_clean = (subject or "").strip() or "Quick Update"
  context_clean = (context or "").strip()

  system_prompt = (
    "You are an expert email copywriter. Write a polished email with this structure:\n"
    "1) greeting\n2) concise purpose\n3) key points\n4) clear call-to-action\n5) polite sign-off.\n"
    f"Tone: {tone_str}. {lang_line} Return only the final email body text, no labels."
  )
  user_prompt = f"Subject: {subject_clean}\nContext/Key points:\n{context_clean}"

  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    # Keep token budget moderate for stability on small local models.
    max_tokens=420,
    temperature=0.65,
  )
  body = _clean_email(raw)
  return {
    "mode": "new_email",
    "subject": subject_clean,
    "tone": tone_str,
    "email": body,
    "word_count": len(re.findall(r"\b[\w'-]+\b", body)),
  }


async def generate_reply_email(
  provider: ModelProvider,
  *,
  original_email: str,
  reply_points: str,
  tone: str | None = None,
  language: str | None = None,
) -> dict[str, Any]:
  tone_str = _normalize_tone(tone)
  lang_line = f"Write in {language}." if language else "Write in clear business English."
  original = (original_email or "").strip()
  points = (reply_points or "").strip()

  system_prompt = (
    "You are an expert assistant for writing email replies. Draft a clear, context-aware "
    "response that addresses the original message and includes all reply points. Keep it "
    f"concise and action-oriented. Tone: {tone_str}. {lang_line} Return only the reply body."
  )
  user_prompt = f"Original email:\n{original}\n\nReply points:\n{points}"

  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    max_tokens=420,
    temperature=0.6,
  )
  body = _clean_email(raw)
  return {
    "mode": "reply",
    "subject": _subject_from_body(body, fallback="Re: Follow-up"),
    "tone": tone_str,
    "email": body,
    "word_count": len(re.findall(r"\b[\w'-]+\b", body)),
  }


async def generate_cold_email(
  provider: ModelProvider,
  *,
  company_name: str,
  purpose_offer: str,
  value_proposition: str,
  tone: str | None = None,
  language: str | None = None,
) -> dict[str, Any]:
  tone_str = _normalize_tone(tone)
  lang_line = f"Write in {language}." if language else "Write in clear business English."
  company = (company_name or "").strip() or "your company"
  purpose = (purpose_offer or "").strip()
  value = (value_proposition or "").strip()

  system_prompt = (
    "You are an expert B2B cold-email copywriter. Write a high-converting cold email:\n"
    "- personalized opener\n- pain point alignment\n- concise value proposition\n"
    "- social-proof style line (non-fabricated generic)\n- clear CTA\n"
    "- brief sign-off\n"
    f"Tone: {tone_str}. {lang_line} Return only the cold email body text."
  )
  user_prompt = (
    f"Target company: {company}\n"
    f"Purpose/Offer: {purpose}\n"
    f"Value proposition: {value}"
  )

  raw = await provider.chat(
    [{"role": "user", "content": user_prompt}],
    system_prompt=system_prompt,
    use_rag=False,
    skip_intent=True,
    max_tokens=420,
    temperature=0.7,
  )
  body = _clean_email(raw)
  return {
    "mode": "cold_email",
    "subject": _subject_from_body(body, fallback=f"Idea for {company}"),
    "tone": tone_str,
    "email": body,
    "word_count": len(re.findall(r"\b[\w'-]+\b", body)),
  }
