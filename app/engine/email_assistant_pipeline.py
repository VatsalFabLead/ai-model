"""Email Assistant — production pipeline v5.2."""

from __future__ import annotations

import re
import time
from typing import Any

from app.engine.email_assistant_enrichment import (
  ARCHITECTURE_FLOW,
  GENERATOR_VERSION,
  adapt_culture,
  build_llm_refinement_prompts,
  build_suggestions,
  check_content_policy,
  classify_business_domain,
  classify_email_subtype,
  classify_relationship,
  compose_email,
  detect_formality,
  detect_intent,
  detect_language,
  detect_sentiment,
  detect_urgency,
  extract_context,
  extract_entities,
  filter_pii,
  generate_alternatives,
  generate_subject_options,
  grammar_check,
  normalize_tone,
  optimize_tone,
  plan_structure,
  professionalism_validate,
  quality_score,
  readability_analyze,
  select_best_subject,
  spam_score,
  style_optimize,
  validate_input,
  analyze_recipient,
)

from app.services.provider_base import ModelProvider


def _clean_llm_email(text: str) -> str:
  text = (text or "").strip()
  text = re.sub(
    r"^(sure[,!.]?\s+)?(here(?:'s| is)|certainly|of course)[^\n:]*:\s*",
    "",
    text,
    flags=re.I,
  )
  text = re.sub(r"^\s*(subject(?:\s*line)?|email|body|reply)\s*:\s*", "", text, flags=re.I)
  text = re.sub(r"^\s*subject(?:\s*line)?\s*[:\-].*\n+", "", text, flags=re.I)
  return re.sub(r"\n{3,}", "\n\n", text).strip()


async def _maybe_refine_with_llm(
  provider: ModelProvider | None,
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
) -> tuple[str, bool]:
  if provider is None:
    return draft, False
  system, user = build_llm_refinement_prompts(
    mode=mode,
    tone=tone,
    subject=subject,
    draft=draft,
    context=context,
    intent=intent,
    email_type=email_type,
    recipient=recipient,
    relationship=relationship,
    domain=domain,
    sentiment=sentiment,
    urgency=urgency,
    culture=culture,
    language=language,
    structure=structure,
  )
  try:
    raw = await provider.chat(
      [{"role": "user", "content": user}],
      system_prompt=system,
      use_rag=False,
      skip_intent=True,
      max_tokens=600,
      temperature=0.45,
    )
    refined = _clean_llm_email(raw)
    return (refined if len(refined) > 50 else draft), True
  except Exception:
    return draft, False


async def run_email_assistant_pipeline(
  mode: str,
  payload: dict[str, Any],
  provider: ModelProvider | None = None,
  *,
  variation_seed: int = 0,
) -> dict[str, Any]:
  t0 = time.perf_counter()
  stages: dict[str, Any] = {}
  tone = normalize_tone(payload.get("tone"))

  stages["input"] = {"mode": mode, "tone": tone}
  validation = validate_input(mode, payload)
  stages["input_validator"] = validation
  if not validation["valid"]:
    raise ValueError("; ".join(validation["issues"]))

  text_blob = " ".join(str(v) for v in payload.values() if isinstance(v, str))
  policy = check_content_policy(text_blob)
  stages["content_policy_gate"] = policy
  if not policy["allowed"]:
    raise ValueError(policy.get("message") or "content_policy_blocked")

  ctx_data = extract_context(mode, payload)
  stages["context_extractor"] = ctx_data
  if mode == "reply":
    stages["thread_parser"] = ctx_data.get("thread", {})

  lang = detect_language(text_blob)
  stages["language_detector"] = lang

  email_type = classify_email_subtype(mode, text_blob)
  stages["email_type_classifier"] = email_type

  intent = detect_intent(text_blob, mode)
  stages["intent_detector"] = intent

  recipient = analyze_recipient(text_blob)
  stages["recipient_analyzer"] = recipient

  relationship = classify_relationship(mode, text_blob)
  stages["relationship_classifier"] = relationship

  entities = extract_entities(text_blob)
  stages["entity_extraction"] = {"entities": entities, "count": len(entities)}

  sentiment = detect_sentiment(text_blob)
  stages["sentiment_detection"] = sentiment

  urgency = detect_urgency(text_blob)
  stages["urgency_detector"] = urgency

  formality = detect_formality(tone, text_blob)
  stages["formality_detector"] = formality

  tone_opt = optimize_tone(
    tone,
    intent["primary_intent"],
    sentiment,
    mode=mode,
    email_type=email_type.get("primary_type", ""),
  )
  effective_tone = tone_opt["effective_tone"]
  stages["tone_optimizer"] = tone_opt

  culture = adapt_culture(lang.get("bcp47", "en"), effective_tone)
  stages["culture_locale_adapter"] = culture

  domain = classify_business_domain(text_blob)
  stages["business_domain_classifier"] = domain

  structure = plan_structure(mode, intent["primary_intent"], effective_tone)
  stages["email_structure_planner"] = {"sections": structure}

  subjects = generate_subject_options(
    ctx_data, mode, intent=intent["primary_intent"], seed=variation_seed,
  )
  stages["subject_generator"] = {"primary": subjects[0], "options": subjects}

  draft = compose_email(
    mode=mode,
    tone=effective_tone,
    context=ctx_data,
    intent=intent["primary_intent"],
    seed=variation_seed,
    sentiment=sentiment,
    urgency=urgency,
    domain=domain,
    culture=culture,
  )
  stages["opening_generator"] = {"mode": "context_aware", "domain_template": domain.get("primary_domain")}
  stages["body_generator"] = {"mode": "template", "word_count": len(draft.split())}
  stages["cta_generator"] = {"intent": intent["primary_intent"]}
  stages["closing_generator"] = {"tone": effective_tone, "culture": culture.get("sign_off")}
  stages["signature_generator"] = {"placeholder": "[Your Name]"}

  email_body, llm_used = await _maybe_refine_with_llm(
    provider,
    mode=mode,
    tone=effective_tone,
    subject=subjects[0],
    draft=draft,
    context=ctx_data,
    intent=intent,
    email_type=email_type,
    recipient=recipient,
    relationship=relationship,
    domain=domain,
    sentiment=sentiment,
    urgency=urgency,
    culture=culture,
    language=lang,
    structure=structure,
  )
  if llm_used:
    stages["body_generator"]["mode"] = "llm_refined"
    stages["body_generator"]["metadata_prompt"] = True

  styled = style_optimize(email_body, effective_tone)
  email_body = styled["text"]
  stages["style_optimizer"] = {"applied": True, "fixes": styled["fixes_applied"], "llm": llm_used}

  grammar = grammar_check(email_body)
  stages["grammar_checker"] = grammar

  readability = readability_analyze(email_body)
  stages["readability_analyzer"] = readability

  subject = select_best_subject(subjects, email_body)
  spam = spam_score(email_body, subject)
  stages["spam_score_checker"] = spam

  pii = filter_pii(email_body)
  if pii["had_pii"]:
    email_body = pii["redacted_text"]
  stages["pii_security_filter"] = {"had_pii": pii["had_pii"], "types": pii["pii_found"]}

  prof = professionalism_validate(email_body, effective_tone)
  stages["professionalism_validator"] = prof

  scores = quality_score(grammar, readability, spam, prof)
  stages["quality_scorer"] = scores

  alternatives = generate_alternatives(
    subject,
    email_body,
    effective_tone,
    mode=mode,
    context=ctx_data,
    intent=intent["primary_intent"],
  )
  stages["alternative_versions"] = {"count": len(alternatives)}

  suggestions = build_suggestions(grammar, prof, spam, readability)
  if lang.get("bcp47", "en") != "en" and not llm_used:
    suggestions.insert(
      0,
      f"Input is {lang.get('language', 'non-English')} — enable AI refinement for a fully localized email body.",
    )

  stages["final_output"] = {
    "subject": subject,
    "word_count": readability["word_count"],
    "reading_time_minutes": readability.get("reading_time_minutes", 1),
    "output_language": lang.get("language", "English"),
  }

  return {
    "generator_version": GENERATOR_VERSION,
    "mode": mode,
    "subject": subject,
    "subject_options": subjects,
    "tone": effective_tone,
    "requested_tone": tone,
    "email": email_body,
    "word_count": readability["word_count"],
    "reading_time_minutes": readability.get("reading_time_minutes", 1),
    "language": lang,
    "output_language": lang.get("language", "English"),
    "quality": scores,
    "scores": {
      "grammar": grammar["score"],
      "readability": readability["reading_ease"],
      "spam_score": spam["spam_score"],
      "spam_risk": spam["spam_risk"],
      "professionalism": prof["score"],
      "overall": scores["overall"],
    },
    "suggestions": suggestions,
    "alternatives": alternatives,
    "architecture": {"flow": ARCHITECTURE_FLOW, "stages": stages},
    "pipeline": {
      "email_type": email_type,
      "intent": intent,
      "recipient": recipient,
      "relationship": relationship,
      "domain": domain,
      "sentiment": sentiment,
      "urgency": urgency,
      "culture": culture,
    },
    "policy": {"status": "allowed"},
    "ai": {"enabled": provider is not None, "model_used": llm_used},
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
  }
