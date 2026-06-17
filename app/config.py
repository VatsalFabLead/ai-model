from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=PROJECT_ROOT / ".env",
    env_file_encoding="utf-8",
    extra="ignore",
  )

  app_name: str = Field(default="Custom Model API", alias="APP_NAME")
  app_env: str = Field(default="dev", alias="APP_ENV")
  api_key: str = Field(default="change-me-to-a-strong-key", alias="API_KEY")
  host: str = Field(default="127.0.0.1", alias="HOST")
  port: int = Field(default=8000, alias="PORT")
  request_timeout_seconds: int = Field(default=120, alias="REQUEST_TIMEOUT_SECONDS")

  # Custom model (100% owned weights — no external model downloads)
  model_id: str = Field(default="custom-nexus-v1", alias="MODEL_ID")
  model_weights_path: Path = Field(
    default=PROJECT_ROOT / "models" / "weights" / "nexus_v1.npz",
    alias="MODEL_WEIGHTS_PATH",
  )
  tokenizer_path: Path = Field(
    default=PROJECT_ROOT / "models" / "tokenizer" / "vocab.json",
    alias="TOKENIZER_PATH",
  )
  corpus_path: Path = Field(
    default=PROJECT_ROOT / "data" / "corpus.txt",
    alias="CORPUS_PATH",
  )
  retrieval_threshold: float = Field(default=0.6, alias="RETRIEVAL_THRESHOLD")

  # Architecture — tuned for Hostinger VPS CPU (increase locally if you have more RAM)
  d_model: int = Field(default=256, alias="D_MODEL")
  n_heads: int = Field(default=4, alias="N_HEADS")
  n_layers: int = Field(default=4, alias="N_LAYERS")
  d_ff: int = Field(default=1024, alias="D_FF")
  max_seq_len: int = Field(default=256, alias="MAX_SEQ_LEN")
  vocab_size: int = Field(default=4096, alias="VOCAB_SIZE")

  # Inference
  max_new_tokens: int = Field(default=128, alias="MAX_NEW_TOKENS")
  temperature: float = Field(default=0.7, alias="TEMPERATURE")
  top_k: int = Field(default=40, alias="TOP_K")
  top_p: float = Field(default=0.9, alias="TOP_P")

  # Security & rate limits (Hostinger-friendly defaults)
  cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
  rate_limit: str = Field(default="60/minute", alias="RATE_LIMIT")
  trusted_hosts: str = Field(default="*", alias="TRUSTED_HOSTS")

  # Merge hook — mount under parent server prefix when integrated
  api_prefix: str = Field(default="/v1", alias="API_PREFIX")
  root_path: str = Field(default="", alias="ROOT_PATH")

  @property
  def is_production(self) -> bool:
    return self.app_env.lower() in {"prod", "production"}

  @property
  def cors_origin_list(self) -> list[str]:
    if self.cors_origins.strip() == "*":
      return ["*"]
    return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
  return Settings()
