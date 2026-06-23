"""Gemma local provider — free, no Hugging Face API.

Loads weights from your folder (e.g. D:\\Gemma 4\\model.safetensors) with
Transformers + PyTorch entirely offline. Falls back to GGUF or Ollama if needed.
Nexus training data is injected as RAG context before generation.
"""

from __future__ import annotations

import asyncio
import gc
from pathlib import Path

import httpx

from app.config import Settings
from app.engine.answer_complete import CONTINUE_PROMPT, is_incomplete_answer, merge_continuation
from app.engine.live_facts import fetch_gold_price_context, is_gold_price_query
from app.engine.resume import detect_resume_intent, generate_resume
from app.engine.tool_context import ToolContextBuilder
from app.services.provider_base import ModelProvider


def _resolve_safetensors_dir(settings: Settings) -> Path | None:
  """Local Transformers folder with config.json + weights (no HF download)."""
  model_dir = Path(settings.gemma_model_dir)
  if not model_dir.is_dir():
    return None
  if not (model_dir / "config.json").is_file():
    return None
  weights_name = settings.gemma_weights_file.strip()
  if weights_name and (model_dir / weights_name).is_file():
    return model_dir
  for name in ("model.safetensors", "pytorch_model.bin"):
    if (model_dir / name).is_file():
      return model_dir
  # Sharded weights
  if list(model_dir.glob("model-*.safetensors")):
    return model_dir
  return None


def _find_gguf(model_dir: Path) -> Path | None:
  if not model_dir.is_dir():
    return None
  for pattern in ("*.gguf", "**/*.gguf"):
    for p in model_dir.glob(pattern):
      if p.is_file() and p.stat().st_size > 1_000_000:
        return p
  return None


def _weights_path(model_dir: Path, settings: Settings) -> Path | None:
  name = settings.gemma_weights_file.strip()
  if name and (model_dir / name).is_file():
    return model_dir / name
  for candidate in ("model.safetensors", "pytorch_model.bin"):
    p = model_dir / candidate
    if p.is_file():
      return p
  return None


def _can_load_safetensors_locally(settings: Settings) -> tuple[bool, str]:
  mode = (settings.gemma_load_safetensors or "auto").lower().strip()
  if mode in ("false", "0", "no", "off"):
    return False, "GEMMA_LOAD_SAFETENSORS=false"
  model_dir = _resolve_safetensors_dir(settings)
  if not model_dir:
    return False, "no local weights found"
  weights = _weights_path(model_dir, settings)
  if not weights:
    return False, "weights file missing"
  if mode in ("true", "1", "yes", "on"):
    return True, "forced"
  need_bytes = int(weights.stat().st_size * 1.2)
  try:
    import psutil

    total = psutil.virtual_memory().total
    if total < need_bytes:
      need_gb = need_bytes / 1e9
      have_gb = total / 1e9
      return False, f"need ~{need_gb:.0f}GB RAM, have {have_gb:.1f}GB — use Ollama or GPU"
  except Exception:
    pass
  return True, "ram ok"


def _read_chat_template(model_dir: Path) -> str:
  tpl = model_dir / "chat_template.jinja"
  if tpl.is_file():
    return tpl.read_text(encoding="utf-8")
  raise FileNotFoundError(
    f"Missing {tpl}. Download once (free) from the Gemma model card "
    "or re-run the app setup — chat_template.jinja is required for text generation."
  )


class GemmaProvider(ModelProvider):
  def __init__(self, settings: Settings) -> None:
    self._settings = settings
    self._ready = False
    self._mode = "ollama"
    self._llm = None
    self._gguf_path: Path | None = None
    self._model = None
    self._processor = None
    self._model_dir: Path | None = None
    self._tools = ToolContextBuilder(settings)

  def _resolve_gguf(self) -> Path | None:
    explicit = self._settings.gemma_gguf_path
    if explicit and Path(explicit).is_file():
      return Path(explicit)
    return _find_gguf(Path(self._settings.gemma_model_dir))

  def _load_safetensors_sync(self) -> None:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_dir = _resolve_safetensors_dir(self._settings)
    if not model_dir:
      raise FileNotFoundError(
        f"No local Gemma weights in {self._settings.gemma_model_dir}. "
        "Expected config.json + model.safetensors"
      )

    device = self._settings.gemma_device.lower().strip()
    if device == "auto":
      device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cpu" else "auto"
    chat_template = _read_chat_template(model_dir)
    self._processor = AutoProcessor.from_pretrained(
      str(model_dir),
      local_files_only=True,
      chat_template=chat_template,
    )
    self._model = AutoModelForMultimodalLM.from_pretrained(
      str(model_dir),
      local_files_only=True,
      dtype=dtype,
      device_map=device,
      low_cpu_mem_usage=True,
    )
    self._model.eval()
    self._model_dir = model_dir
    self._tools.load_kb()
    self._mode = "safetensors"
    self._ready = True

  def _load_gguf_sync(self) -> None:
    from llama_cpp import Llama

    path = self._resolve_gguf()
    if not path:
      raise FileNotFoundError("No GGUF found in Gemma model directory")
    self._gguf_path = path
    self._llm = Llama(
      model_path=str(path),
      n_ctx=self._settings.llm_context,
      n_threads=self._settings.llm_threads,
      n_gpu_layers=self._settings.llm_gpu_layers,
      verbose=False,
    )
    self._tools.load_kb()
    self._mode = "gguf"
    self._ready = True

  async def load(self) -> None:
    can_load, reason = _can_load_safetensors_locally(self._settings)
    if can_load:
      try:
        await asyncio.to_thread(self._load_safetensors_sync)
        return
      except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
          "Gemma safetensors load failed (%s); trying GGUF/Ollama fallback", exc
        )
    elif _resolve_safetensors_dir(self._settings):
      import logging

      logging.getLogger(__name__).info(
        "Skipping direct model.safetensors load (%s); using fallback backend", reason
      )

    gguf = self._resolve_gguf()
    if gguf:
      await asyncio.to_thread(self._load_gguf_sync)
      return

    base = self._settings.ollama_host.rstrip("/")
    model = self._settings.gemma_ollama_model
    try:
      async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{base}/api/tags")
        r.raise_for_status()
    except httpx.HTTPError as exc:
      raise RuntimeError(
        f"No local Gemma weights loaded and Ollama unreachable at {base}.\n"
        f"Place model.safetensors in {self._settings.gemma_model_dir} "
        f"or run: ollama pull {model}\n"
        f"Original error: {exc}"
      ) from exc

    self._tools.load_kb()
    self._mode = "ollama"
    self._ready = True

  async def unload(self) -> None:
    self._llm = None
    self._model = None
    self._processor = None
    self._ready = False
    gc.collect()
    try:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

  def is_ready(self) -> bool:
    return self._ready

  async def _chat_ollama(
    self,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
  ) -> str:
    base = self._settings.ollama_host.rstrip("/")
    max_passes = max(1, self._settings.chat_completion_max_passes)
    answer = ""
    for attempt in range(max_passes):
      run_messages = list(messages)
      if attempt > 0 and answer.strip():
        run_messages = run_messages + [
          {"role": "assistant", "content": answer.strip()},
          {"role": "user", "content": CONTINUE_PROMPT},
        ]
      payload = {
        "model": self._settings.gemma_ollama_model,
        "messages": run_messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
      }
      async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
      chunk = (data.get("message", {}).get("content") or "").strip()
      answer = merge_continuation(answer, chunk)
      if data.get("done", True) and not is_incomplete_answer(answer):
        break
    return answer

  def _chat_gguf_sync(
    self,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
  ) -> str:
    resp = self._llm.create_chat_completion(
      messages=messages,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=self._settings.llm_top_p,
      top_k=self._settings.llm_top_k,
      repeat_penalty=self._settings.llm_repeat_penalty,
    )
    return (resp["choices"][0].get("message", {}).get("content") or "").strip()

  def _chat_safetensors_sync(
    self,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
  ) -> str:
    import torch

    if not self._model or not self._processor:
      raise RuntimeError("Gemma safetensors model is not loaded")

    chat: list[dict[str, str]] = []
    for m in messages:
      role = m.get("role", "user")
      if role in ("system", "user", "assistant"):
        chat.append({"role": role, "content": m.get("content", "")})

    inputs = self._processor.apply_chat_template(
      chat,
      tokenize=True,
      return_dict=True,
      return_tensors="pt",
      add_generation_prompt=True,
      enable_thinking=False,
    )
    device = next(self._model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs: dict = {
      "max_new_tokens": max_tokens,
      "do_sample": temperature > 0,
      "top_p": self._settings.llm_top_p,
      "top_k": self._settings.llm_top_k,
    }
    if temperature > 0:
      gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
      outputs = self._model.generate(**inputs, **gen_kwargs)

    raw = self._processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    if hasattr(self._processor, "parse_response"):
      try:
        parsed = self._processor.parse_response(raw)
        if isinstance(parsed, str):
          return parsed.strip()
        if isinstance(parsed, dict):
          return str(parsed.get("content") or parsed.get("text") or raw).strip()
      except Exception:
        pass
    return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

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

    system_prompt = kwargs.get("system_prompt") or self._settings.gemma_system_prompt
    use_rag = kwargs.get("use_rag")
    if use_rag is None:
      use_rag = self._settings.use_rag

    full_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if use_rag and last_user.strip():
      context = await self._tools.gather(last_user)
      if context:
        full_messages.append({
          "role": "system",
          "content": f"Nexus training context (use if relevant):\n{context[:28000]}",
        })
    full_messages.extend(messages)

    temperature = float(kwargs.get("temperature") or self._settings.gemma_temperature)
    max_tokens = int(kwargs.get("max_tokens") or self._settings.gemma_max_tokens)

    if self._mode == "safetensors" and self._model:
      return await asyncio.to_thread(
        self._chat_safetensors_sync,
        full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
      )
    if self._mode == "gguf" and self._llm:
      return await asyncio.to_thread(
        self._chat_gguf_sync,
        full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
      )
    return await self._chat_ollama(full_messages, temperature=temperature, max_tokens=max_tokens)

  def model_id(self) -> str:
    if self._mode == "safetensors" and self._model_dir:
      return f"gemma/{self._model_dir.name}/model.safetensors"
    if self._mode == "gguf" and self._gguf_path:
      return f"gemma/{self._gguf_path.name}"
    return f"gemma/{self._settings.gemma_ollama_model}"
