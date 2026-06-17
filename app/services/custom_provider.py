import asyncio
from pathlib import Path

from app.config import Settings
from app.engine.inference import GenerationConfig, InferenceEngine
from app.engine.knowledge import KnowledgeBase, load_knowledge_base
from app.engine.retriever import KnowledgeRetriever, load_retriever
from app.engine.web_knowledge import WikipediaSource
from app.services.provider_base import ModelProvider


class CustomModelProvider(ModelProvider):
  """Serves the fully custom stack — 100% custom, free, no third-party AI models.

  Answering order:
    1) Custom TF-IDF knowledge base (your own data; instant; multilingual)
    2) Free encyclopedia source (Wikipedia) for detailed world knowledge
    3) Graceful fallback (or optional neural generation)
  """

  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._engine = InferenceEngine(settings)
    self._retriever: KnowledgeRetriever | None = None
    self._kb: KnowledgeBase | None = None
    self._wiki: WikipediaSource | None = None
    if settings.enable_web_knowledge:
      self._wiki = WikipediaSource(
        sentences=settings.web_knowledge_sentences,
        timeout=settings.web_knowledge_timeout,
      )

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

  _FALLBACK = (
    "I don't have a confident answer for that yet. You can teach me by adding it "
    "to data/knowledge.jsonl (as {\"q\": \"...\", \"a\": \"...\"}) or by importing a "
    "free dataset with scripts/import_dataset.py, then redeploying."
  )

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    last_user = next(
      (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
      "",
    )

    if last_user.strip() and self._kb and self._kb.size:
      answer, score = self._kb.search(last_user)
      if answer and score >= self._settings.knowledge_threshold:
        return answer

    if last_user.strip() and self._wiki is not None:
      wiki_answer = await self._wiki.query(last_user)
      if wiki_answer:
        return wiki_answer

    if kwargs.get("use_neural_fallback"):
      prompt = self._engine.format_chat_prompt(messages)
      config = GenerationConfig(
        max_new_tokens=int(kwargs.get("max_tokens", self._settings.max_new_tokens)),
        temperature=float(kwargs.get("temperature", self._settings.temperature)),
        top_k=int(kwargs.get("top_k", self._settings.top_k)),
        top_p=float(kwargs.get("top_p", self._settings.top_p)),
      )
      return await self._engine.generate_async(prompt, config)

    return self._FALLBACK

  def model_id(self) -> str:
    return self._engine.model_id
