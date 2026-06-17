import asyncio
from pathlib import Path

from app.config import Settings
from app.engine.inference import GenerationConfig, InferenceEngine
from app.engine.retriever import KnowledgeRetriever, load_retriever
from app.services.provider_base import ModelProvider


class CustomModelProvider(ModelProvider):
  """Serves the fully custom NumPy transformer with an embedding retrieval layer.

  Both the generator and the retriever are 100% custom and free — no third-party
  models. The retriever uses the trained model's own learned embeddings.
  """

  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._engine = InferenceEngine(settings)
    self._retriever: KnowledgeRetriever | None = None

  def _build_retriever(self) -> None:
    model = self._engine.model
    tokenizer = self._engine.tokenizer
    if model is not None and tokenizer is not None:
      self._retriever = load_retriever(model, tokenizer, Path(self._settings.corpus_path))

  async def load(self) -> None:
    await asyncio.to_thread(self._engine.load)
    await asyncio.to_thread(self._build_retriever)

  async def unload(self) -> None:
    self._engine.unload()
    self._retriever = None

  def is_ready(self) -> bool:
    return self._engine.is_ready()

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    if self._retriever and last_user.strip():
      answer, score = self._retriever.retrieve(last_user)
      if answer and score >= self._settings.retrieval_threshold:
        return answer

    prompt = self._engine.format_chat_prompt(messages)
    config = GenerationConfig(
      max_new_tokens=int(kwargs.get("max_tokens", self._settings.max_new_tokens)),
      temperature=float(kwargs.get("temperature", self._settings.temperature)),
      top_k=int(kwargs.get("top_k", self._settings.top_k)),
      top_p=float(kwargs.get("top_p", self._settings.top_p)),
    )
    return await self._engine.generate_async(prompt, config)

  def model_id(self) -> str:
    return self._engine.model_id
