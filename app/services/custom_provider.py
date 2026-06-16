from app.config import Settings
from app.engine.inference import GenerationConfig, InferenceEngine
from app.services.provider_base import ModelProvider


class CustomModelProvider(ModelProvider):
  """Serves the fully custom NumPy transformer."""

  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._engine = InferenceEngine(settings)

  async def load(self) -> None:
    await __import__("asyncio").to_thread(self._engine.load)

  async def unload(self) -> None:
    self._engine.unload()

  def is_ready(self) -> bool:
    return self._engine.is_ready()

  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
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
