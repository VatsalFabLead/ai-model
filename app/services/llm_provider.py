"""Free open-source local LLM provider (GGUF via llama.cpp).

This runs an OPEN-SOURCE model on your own machine/server. It is NOT GPT/Claude/
Gemini and uses no external AI company or API. Optional RAG context is drawn from
your custom knowledge base, live facts, and free Wikipedia.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.engine.answer_complete import CONTINUE_PROMPT, is_incomplete_answer, merge_continuation
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.tool_context import ToolContextBuilder
from app.services.provider_base import ModelProvider


class LocalLLMProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._llm = None
    self._tools = ToolContextBuilder(settings)

  def _load_sync(self) -> None:
    from llama_cpp import Llama

    model_path = Path(self._settings.llm_model_path)
    if not model_path.exists():
      raise FileNotFoundError(
        f"LLM model not found at {model_path}. Download it first:\n"
        f"  python scripts/download_model.py"
      )
    self._llm = Llama(
      model_path=str(model_path),
      n_ctx=self._settings.llm_context,
      n_threads=self._settings.llm_threads,
      n_gpu_layers=self._settings.llm_gpu_layers,
      verbose=False,
    )
    self._tools.load_kb()

  async def load(self) -> None:
    await asyncio.to_thread(self._load_sync)

  async def unload(self) -> None:
    self._llm = None

  def is_ready(self) -> bool:
    return self._llm is not None

  def _generate_sync(
    self,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
  ) -> tuple[str, str]:
    resp = self._llm.create_chat_completion(
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=self._settings.llm_top_p,
      top_k=self._settings.llm_top_k,
      repeat_penalty=self._settings.llm_repeat_penalty,
      frequency_penalty=self._settings.llm_frequency_penalty,
      presence_penalty=self._settings.llm_presence_penalty,
    )
    choice = resp["choices"][0]
    text = (choice.get("message", {}).get("content") or "").strip()
    finish = str(choice.get("finish_reason") or "")
    return text, finish

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

    system_prompt = kwargs.get("system_prompt") or self._settings.llm_system_prompt
    use_rag = kwargs.get("use_rag")
    if use_rag is None:
      use_rag = self._settings.use_rag

    base_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if use_rag and last_user.strip():
      context = await self._tools.gather(last_user)
      if context:
        base_messages.append({
          "role": "system",
          "content": f"Reference context (use if relevant):\n{context[:28000]}",
        })

    base_messages.extend(messages)

    max_tokens = int(kwargs.get("max_tokens") or self._settings.llm_max_tokens)
    temperature = kwargs.get("temperature")
    temperature = float(temperature if temperature is not None else self._settings.llm_temperature)
    max_passes = max(1, self._settings.chat_completion_max_passes)

    answer = ""
    for attempt in range(max_passes):
      run_messages = list(base_messages)
      if attempt > 0 and answer.strip():
        run_messages = run_messages + [
          {"role": "assistant", "content": answer.strip()},
          {"role": "user", "content": CONTINUE_PROMPT},
        ]

      text, finish = await asyncio.to_thread(
        self._generate_sync, run_messages, max_tokens, temperature
      )
      answer = merge_continuation(answer, text)
      if finish != "length" and not is_incomplete_answer(answer):
        break

    return answer

  def model_id(self) -> str:
    return f"llm/{self._settings.llm_model_path.name}"
