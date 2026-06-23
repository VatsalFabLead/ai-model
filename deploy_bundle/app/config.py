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
  # One or more valid API keys (comma-separated). Generate with scripts/generate_key.py.
  # Each client/app can have its own key so you can revoke them individually.
  api_key: str = Field(default="change-me-to-a-strong-key", alias="API_KEY")
  host: str = Field(default="127.0.0.1", alias="HOST")
  port: int = Field(default=8000, alias="PORT")
  request_timeout_seconds: int = Field(default=120, alias="REQUEST_TIMEOUT_SECONDS")

  # Provider backend: "custom" (from-scratch NumPy), "llm" (GGUF via llama.cpp),
  # or "ollama" (free open-source models via local Ollama server).
  model_backend: str = Field(default="custom", alias="MODEL_BACKEND")

  # Ollama (free, open-source local runtime). Not GPT/Claude/Gemini.
  ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
  ollama_model: str = Field(default="qwen2.5:0.5b", alias="OLLAMA_MODEL")

  # Free open-source local model (GGUF via llama.cpp). Not GPT/Claude/Gemini.
  llm_model_path: Path = Field(
    default=PROJECT_ROOT / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    alias="LLM_MODEL_PATH",
  )
  llm_context: int = Field(default=4096, alias="LLM_CONTEXT")
  llm_threads: int = Field(default=4, alias="LLM_THREADS")
  llm_gpu_layers: int = Field(default=0, alias="LLM_GPU_LAYERS")
  llm_max_tokens: int = Field(default=512, alias="LLM_MAX_TOKENS")
  llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
  # Anti-repetition sampling (important for small models to avoid loops).
  llm_top_p: float = Field(default=0.95, alias="LLM_TOP_P")
  llm_top_k: int = Field(default=40, alias="LLM_TOP_K")
  llm_repeat_penalty: float = Field(default=1.3, alias="LLM_REPEAT_PENALTY")
  llm_frequency_penalty: float = Field(default=0.5, alias="LLM_FREQUENCY_PENALTY")
  llm_presence_penalty: float = Field(default=0.3, alias="LLM_PRESENCE_PENALTY")
  llm_system_prompt: str = Field(
    default=(
      "You are Nexus, a helpful, detailed, and friendly AI assistant. "
      "Answer clearly and thoroughly using well-structured markdown. "
      "When given context, use it; if you are unsure, say so honestly."
    ),
    alias="LLM_SYSTEM_PROMPT",
  )
  use_rag: bool = Field(default=True, alias="USE_RAG")

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
  knowledge_path: Path = Field(
    default=PROJECT_ROOT / "data" / "knowledge.jsonl",
    alias="KNOWLEDGE_PATH",
  )
  retrieval_threshold: float = Field(default=0.6, alias="RETRIEVAL_THRESHOLD")
  knowledge_threshold: float = Field(default=0.18, alias="KNOWLEDGE_THRESHOLD")

  # Free encyclopedia source (Wikipedia) for detailed world knowledge.
  # This is a data source, NOT an AI model. Set to false to stay fully offline.
  enable_web_knowledge: bool = Field(default=True, alias="ENABLE_WEB_KNOWLEDGE")
  web_knowledge_sentences: int = Field(default=8, alias="WEB_KNOWLEDGE_SENTENCES")
  web_knowledge_timeout: float = Field(default=8.0, alias="WEB_KNOWLEDGE_TIMEOUT")

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
  # Set RATE_LIMIT_ENABLED=false for no rate limit. When enabled, RATE_LIMIT is the
  # max requests per IP (slowapi format, e.g. "300/minute", "10/second").
  rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
  rate_limit: str = Field(default="300/minute", alias="RATE_LIMIT")
  trusted_hosts: str = Field(default="*", alias="TRUSTED_HOSTS")

  # Merge hook — mount under parent server prefix when integrated
  api_prefix: str = Field(default="/v1", alias="API_PREFIX")
  root_path: str = Field(default="", alias="ROOT_PATH")

  # Dev auto-reload (off by default: avoids reloading the LLM on every edit)
  reload: bool = Field(default=False, alias="RELOAD")

  @property
  def is_production(self) -> bool:
    return self.app_env.lower() in {"prod", "production"}

  @property
  def api_key_list(self) -> list[str]:
    return [k.strip() for k in self.api_key.split(",") if k.strip()]

  @property
  def cors_origin_list(self) -> list[str]:
    if self.cors_origins.strip() == "*":
      return ["*"]
    return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
  return Settings()
