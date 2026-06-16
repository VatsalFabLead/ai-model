from abc import ABC, abstractmethod


class ModelProvider(ABC):
  """Interface for merging this service into a larger server."""

  @abstractmethod
  async def load(self) -> None:
    pass

  @abstractmethod
  async def unload(self) -> None:
    pass

  @abstractmethod
  def is_ready(self) -> bool:
    pass

  @abstractmethod
  async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
    pass

  @abstractmethod
  def model_id(self) -> str:
    pass
