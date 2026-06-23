"""Custom model provider — inference-only (no direct dataset reads as answers).

Training happens beforehand via scripts/train.py. At chat time Nexus uses:
  1. Model weights (knowledge from training)
  2. Conversation history
  3. Optional tools (KB, embeddings, Wikipedia) as RAG context only

No GPT, Claude, or Gemini.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from app.config import Settings
from app.engine.inference import GenerationConfig, InferenceEngine
from app.engine.knowledge import KnowledgeBase, load_knowledge_base, tokenize
from app.engine.knowledge import _STOPWORDS
from app.engine.answer_format import count_words, format_answer_text, strip_source_attribution
from app.engine.answer_complete import CONTINUE_PROMPT, is_incomplete_answer, merge_continuation
from app.engine.assistant_profile import inference_system_prompt
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.retriever import KnowledgeRetriever, load_retriever
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.engine.tool_context import ToolContextBuilder
from app.services.backend_router import is_low_quality_output
from app.services.provider_base import ModelProvider

_CHAT_SYSTEM = inference_system_prompt()


def _relevant(query: str, text: str) -> bool:
  q_words = set(tokenize(query)) - _STOPWORDS
  if not q_words:
    return True
  return not q_words.isdisjoint(set(tokenize(text)))


def _is_assistant_meta_question(text: str) -> bool:
  low = text.strip().lower()
  meta = (
    "who are you", "what are you", "what can you do", "your capabilities",
    "your rules", "your personality", "what rules do you", "how should you format",
    "what if you are unsure", "nexus", "custom model", "custom ai",
  )
  return any(m in low for m in meta)


def _is_detailed_topic(text: str) -> bool:
  t = text.strip()
  if re.match(
    r"^(what|who|where|when|why|how|which|define|explain|tell me about)\b",
    t,
    flags=re.IGNORECASE,
  ):
    return True
  words = re.findall(r"\w+", t, flags=re.UNICODE)
  return 1 <= len(words) <= 4


def _query_variants(text: str) -> list[str]:
  q = (text or "").strip()
  if not q:
    return []
  variants = [q]
  short = re.sub(
    r"^(what|who|where|when|why|how|which|tell me about|explain|define|please)\b[\s:,]*",
    "",
    q,
    flags=re.IGNORECASE,
  ).strip(" ?.!,")
  if short and short.lower() != q.lower():
    variants.append(short)
  return variants


def _is_prediction_question(text: str) -> bool:
  return bool(
    re.search(
      r"\b(predict|prediction|forecast|price after|after \d+ years?|in \d+ years?|"
      r"will .+ (rise|fall|be|cost)|future price)\b",
      text,
      flags=re.IGNORECASE,
    )
  )


def _wants_long_answer(text: str) -> bool:
  low = text.lower()
  if any(
    m in low
    for m in (
      "guide", "comprehensive", "overview", "in detail", "step by step",
      "key takeaway", "practical example", "beginner", "complete answer",
      "full explanation", "write an article", "write a blog",
    )
  ):
    return True
  return _is_detailed_topic(text) or _is_prediction_question(text)


def _inference_guidance(question: str) -> str:
  if _is_prediction_question(question):
    return (
      "The user asks about a future prediction. You cannot know exact future prices. "
      "Answer with inference only: ## Short answer, ## Key factors, ## Historical context, "
      "## Uncertainty, ## Not financial advice. Use clear headings. Never cite sources."
    )
  return (
    "Answer the user through your trained inference. Use headings and bullets. "
    "Synthesize reference material in your own words. Never cite sources or paste verbatim. "
    "Always finish every section, code block, and summary — do not stop mid-heading."
  )


def _weak_generation(text: str, *, detailed: bool = False) -> bool:
  t = (text or "").strip()
  if len(t) < 15:
    return True
  low = t.lower()
  if "don't have a confident answer" in low:
    return True
  if "knowledge.jsonl" in low or "import_dataset" in low:
    return True
  if "source: wikipedia" in low or "_(source:" in low:
    return True
  if low.startswith("user:") or low.count("user:") >= 2:
    return True
  min_words = 20 if detailed else 8
  if count_words(t) < min_words:
    return True
  return False


def _effective_max_tokens(kwargs: dict, settings: Settings) -> int:
  requested = int(kwargs.get("max_tokens") or settings.max_new_tokens)
  return max(requested, settings.max_new_tokens)


def _trim_tool_context(context: str, max_chars: int) -> str:
  ctx = strip_source_attribution(context).strip()
  if len(ctx) <= max_chars:
    return ctx
  return ctx[: max_chars - 20].rstrip() + "\n...[context trimmed]"


def _format_answer(settings: Settings, text: str, *, detailed: bool) -> str:
  return format_answer_text(
    text,
    min_words=settings.answer_min_words if detailed else 0,
    max_words=settings.answer_max_words,
  )


class CustomModelProvider(ModelProvider):
  """Inference-only custom stack: tools feed context; the transformer writes every reply."""

  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._engine = InferenceEngine(settings)
    self._retriever: KnowledgeRetriever | None = None
    self._kb: KnowledgeBase | None = None
    self._wiki = None
    self._tools = ToolContextBuilder(settings)
    if settings.enable_web_knowledge:
      from app.engine.web_knowledge import WikipediaSource

      self._wiki = WikipediaSource(
        min_words=0,
        max_words=settings.tool_context_max_words,
        timeout=settings.web_knowledge_timeout,
      )
      self._tools._wiki = self._wiki

  def _build_retriever(self) -> None:
    model = self._engine.model
    tokenizer = self._engine.tokenizer
    if model is not None and tokenizer is not None:
      self._retriever = load_retriever(model, tokenizer, Path(self._settings.corpus_path))
    self._kb = load_knowledge_base(
      knowledge_path=Path(self._settings.knowledge_path),
      corpus_path=Path(self._settings.corpus_path),
    )

  async def load(self) -> None:
    await asyncio.to_thread(self._engine.load)
    await asyncio.to_thread(self._build_retriever)

  async def unload(self) -> None:
    self._engine.unload()
    self._retriever = None
    self._kb = None

  def is_ready(self) -> bool:
    return self._engine.is_ready()

  async def _gather_tool_context(self, query: str) -> str:
    if self._kb is None:
      self._kb = load_knowledge_base(
        knowledge_path=Path(self._settings.knowledge_path),
        corpus_path=Path(self._settings.corpus_path),
      )
    self._tools.load_kb(self._retriever)
    return await self._tools.gather(query)

  async def _run_inference(
    self,
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
    tool_context: str,
    max_new_tokens: int,
    temperature: float,
    **kwargs,
  ) -> str:
    tool_block = _trim_tool_context(tool_context, self._settings.inference_context_chars)
    full_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if tool_block:
      full_messages.append({
        "role": "system",
        "content": (
          "Reference material from tools (synthesize in your own words; "
          "do not cite sources or paste verbatim):\n"
          f"{tool_block}"
        ),
      })
    full_messages.extend(messages)
    prompt = self._engine.format_chat_prompt(full_messages)
    gen_kwargs = {
      k: v for k, v in kwargs.items()
      if k not in ("max_tokens", "temperature", "system_prompt", "tool_context", "detailed")
    }
    config = GenerationConfig(
      max_new_tokens=max_new_tokens,
      temperature=temperature,
      top_k=int(gen_kwargs.get("top_k", self._settings.top_k)),
      top_p=float(gen_kwargs.get("top_p", self._settings.top_p)),
    )
    return await self._engine.generate_async(prompt, config)

  async def _generate_answer(
    self,
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
    tool_context: str,
    detailed: bool,
    temperature: float | None = None,
    **kwargs,
  ) -> str:
    if not self._engine.is_ready():
      raise RuntimeError("Custom model weights are not loaded. Run scripts/train.py first.")

    base_tokens = _effective_max_tokens(kwargs, self._settings)
    passes = self._settings.inference_max_passes if detailed else 1
    if detailed:
      passes = max(passes, 4)
    answer = ""
    base_temp = float(
      temperature if temperature is not None else kwargs.get("temperature", self._settings.temperature)
    )

    for attempt in range(passes):
      temp = max(0.35, base_temp - 0.1 * attempt) if attempt > 0 else base_temp

      chunk_messages = list(messages)
      if answer.strip():
        chunk_messages = chunk_messages + [
          {"role": "assistant", "content": answer.strip()},
          {"role": "user", "content": CONTINUE_PROMPT if attempt > 0 else "Continue your answer with more detail, same style and language."},
        ]

      chunk = await self._run_inference(
        chunk_messages,
        system_prompt=system_prompt,
        tool_context=tool_context if attempt == 0 else "",
        max_new_tokens=base_tokens,
        temperature=temp,
        **kwargs,
      )
      chunk = strip_source_attribution(chunk).strip()
      if not chunk:
        break
      answer = merge_continuation(answer, chunk)

      if not detailed:
        break
      if self._settings.answer_min_words > 0 and count_words(answer) >= self._settings.answer_min_words:
        if not is_incomplete_answer(answer):
          break
      if not is_incomplete_answer(answer):
        if not _weak_generation(chunk, detailed=detailed):
          break
      if _weak_generation(chunk, detailed=detailed) and attempt >= passes - 1:
        break

    return answer

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    if not kwargs.get("skip_intent") and detect_resume_intent(last_user):
      return generate_resume(last_user)

    if is_gold_price_query(last_user):
      live = await fetch_gold_price_context(last_user)
      if live:
        return live

    system_prompt = (kwargs.get("system_prompt") or "").strip() or _CHAT_SYSTEM
    domain_context = (kwargs.get("domain_context") or "").strip()

    if kwargs.get("skip_intent"):
      infer_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("system_prompt", "domain_context", "skip_intent", "use_rag")
      }
      infer_kwargs["max_tokens"] = _effective_max_tokens(kwargs, self._settings)
      try:
        generated = await self._generate_answer(
          messages,
          system_prompt=system_prompt,
          tool_context=domain_context,
          detailed=False,
          **infer_kwargs,
        )
      except RuntimeError as exc:
        return str(exc)
      if generated and not is_low_quality_output(generated):
        return strip_source_attribution(generated).strip()
      raise RuntimeError("Model could not produce a valid rewrite.")

    detailed = _wants_long_answer(last_user)
    guidance = _inference_guidance(last_user)
    full_system = f"{system_prompt}\n\n{guidance}"

    infer_kwargs = {
      k: v for k, v in kwargs.items()
      if k not in ("system_prompt", "domain_context", "skip_intent")
    }
    infer_kwargs["max_tokens"] = _effective_max_tokens(kwargs, self._settings)

    use_rag = kwargs.get("use_rag")
    if use_rag is None:
      use_rag = self._settings.use_rag

    tool_context = domain_context
    if use_rag and last_user.strip():
      dynamic = await self._gather_tool_context(last_user)
      if dynamic:
        tool_context = f"{tool_context}\n\n{dynamic}".strip() if tool_context else dynamic

    try:
      generated = await self._generate_answer(
        messages,
        system_prompt=full_system,
        tool_context=tool_context,
        detailed=detailed,
        **infer_kwargs,
      )
    except RuntimeError as exc:
      return str(exc)

    if generated and not _weak_generation(generated, detailed=detailed) and not is_low_quality_output(generated):
      return _format_answer(self._settings, generated, detailed=detailed)

    focused = [{"role": "user", "content": last_user}]
    retry = await self._generate_answer(
      focused,
      system_prompt=full_system,
      tool_context=tool_context,
      detailed=detailed,
      max_tokens=infer_kwargs["max_tokens"] + 64,
      temperature=0.45,
    )
    if retry and not _weak_generation(retry, detailed=detailed) and not is_low_quality_output(retry):
      return _format_answer(self._settings, retry, detailed=detailed)

    return (
      "I could not produce a confident answer through inference yet. "
      "Please run: python scripts/train_worldwide.py --epochs 12\n"
      "Then restart the server with: python run.py"
    )

  def model_id(self) -> str:
    return self._engine.model_id
