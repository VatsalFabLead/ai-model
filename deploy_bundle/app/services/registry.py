from app.config import Settings
from app.services.custom_provider import CustomModelProvider
from app.services.provider_base import ModelProvider


class ProviderRegistry:
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._provider: ModelProvider | None = None

  def _create_provider(self) -> ModelProvider:
    backend = self._settings.model_backend.lower().strip()
    if backend == "ollama":
      from app.services.ollama_provider import OllamaProvider

      return OllamaProvider(self._settings)
    if backend == "llm":
      from app.services.llm_provider import LocalLLMProvider

      return LocalLLMProvider(self._settings)
    return CustomModelProvider(self._settings)

  async def startup(self) -> None:
    self._provider = self._create_provider()
    await self._provider.load()

  async def shutdown(self) -> None:
    if self._provider:
      await self._provider.unload()
    self._provider = None

  @property
  def provider(self) -> ModelProvider:
    if not self._provider:
      raise RuntimeError("Provider registry is not initialized")
    return self._provider

  def is_ready(self) -> bool:
    return self._provider is not None and self._provider.is_ready()
