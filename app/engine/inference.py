"""Inference engine for the custom transformer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.engine.tokenizer import ByteTokenizer
from app.engine.transformer import CustomTransformer, ModelConfig


@dataclass
class GenerationConfig:
  max_new_tokens: int = 256
  temperature: float = 0.7
  top_k: int = 40
  top_p: float = 0.9
  stop_tokens: tuple[str, ...] = ("<|eos|>",)


class InferenceEngine:
  """Runs custom model inference — async-friendly for FastAPI."""

  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._tokenizer: ByteTokenizer | None = None
    self._model: CustomTransformer | None = None

  def is_ready(self) -> bool:
    return self._model is not None and self._tokenizer is not None

  def load(self) -> None:
    weights_path = Path(self._settings.model_weights_path)
    tokenizer_path = Path(self._settings.tokenizer_path)

    if weights_path.exists():
      self._model = CustomTransformer.load(weights_path)
    else:
      config = ModelConfig(
        vocab_size=self._settings.vocab_size,
        d_model=self._settings.d_model,
        n_heads=self._settings.n_heads,
        n_layers=self._settings.n_layers,
        d_ff=self._settings.d_ff,
        max_seq_len=self._settings.max_seq_len,
      )
      self._model = CustomTransformer(config)

    if tokenizer_path.exists():
      self._tokenizer = ByteTokenizer.load(tokenizer_path)
    else:
      self._tokenizer = ByteTokenizer.train_from_text(
        "Hello world. This is a custom model.",
        vocab_size=self._settings.vocab_size,
      )
      self._tokenizer.save(tokenizer_path)

  def unload(self) -> None:
    self._model = None
    self._tokenizer = None

  def _sample_next(self, logits: np.ndarray, config: GenerationConfig) -> int:
    logits = logits.astype(np.float64)
    if config.temperature <= 0:
      return int(np.argmax(logits))

    logits = logits / max(config.temperature, 1e-6)

    if config.top_k > 0:
      top_idx = np.argpartition(logits, -config.top_k)[-config.top_k :]
      mask = np.full_like(logits, -1e9)
      mask[top_idx] = logits[top_idx]
      logits = mask

    if 0 < config.top_p < 1.0:
      sorted_idx = np.argsort(logits)[::-1]
      sorted_logits = logits[sorted_idx]
      probs = np.exp(sorted_logits - np.max(sorted_logits))
      probs = probs / probs.sum()
      cumulative = np.cumsum(probs)
      keep = cumulative <= config.top_p
      keep[0] = True
      filtered = np.full_like(logits, -1e9)
      filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
      logits = filtered

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

  def generate(
    self,
    prompt: str,
    config: GenerationConfig | None = None,
  ) -> str:
    if not self._model or not self._tokenizer:
      raise RuntimeError("Inference engine is not loaded")

    gen_cfg = config or GenerationConfig(
      max_new_tokens=self._settings.max_new_tokens,
      temperature=self._settings.temperature,
      top_k=self._settings.top_k,
      top_p=self._settings.top_p,
    )

    eos_id = self._tokenizer._vocab.get(ByteTokenizer.EOS, -1)
    context = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated = list(context)

    for _ in range(gen_cfg.max_new_tokens):
      seq = np.array([generated[-self._settings.max_seq_len :]], dtype=np.int32)
      logits = self._model.forward(seq)
      next_id = self._sample_next(logits[0, -1], gen_cfg)
      generated.append(next_id)
      if next_id == eos_id:
        break

    return self._tokenizer.decode(generated[len(context) :])

  async def generate_async(
    self,
    prompt: str,
    config: GenerationConfig | None = None,
  ) -> str:
    return await asyncio.to_thread(self.generate, prompt, config)

  def format_chat_prompt(self, messages: list[dict[str, str]]) -> str:
    """Convert OpenAI-style messages into model prompt."""
    parts: list[str] = []
    for msg in messages:
      role = msg.get("role", "user").strip().lower()
      content = msg.get("content", "").strip()
      if not content:
        continue
      if role == "system":
        parts.append(f"System: {content}\n")
      elif role == "assistant":
        parts.append(f"Assistant: {content}\n")
      else:
        parts.append(f"User: {content}\n")
    parts.append("Assistant: ")
    return "".join(parts)

  @property
  def model_id(self) -> str:
    return self._settings.model_id

  @property
  def tokenizer(self) -> ByteTokenizer | None:
    return self._tokenizer

  @property
  def model(self) -> CustomTransformer | None:
    return self._model
