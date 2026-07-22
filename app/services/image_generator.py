"""Image Generator Service Layer — prompt enhancement & custom model execution."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from app.engine.image_generator_engine import (
  generate_image_matrix,
  get_available_styles,
  image_to_base64,
)
from app.services.provider_base import ModelProvider

logger = logging.getLogger("uvicorn.error")


async def enhance_prompt(
  provider: Optional[ModelProvider],
  prompt: str,
  style: str,
) -> str:
  """Use ChatGPT DALL-E 3 style prompt expansion to enrich raw prompts into studio masterworks."""
  raw = prompt.strip()
  if not raw:
    return raw

  if provider and hasattr(provider, "generate"):
    system_prompt = (
      "You are an expert ChatGPT DALL-E 3 Image Prompt Engineer. Expand the user's prompt into a vivid, "
      "photorealistic visual description. Include exact lighting, camera lens specs (35mm f/1.8), atmosphere, "
      "volumetric fog, ray tracing, textures, materials, and hyper-detailed composition. "
      "Keep under 80 words, focused purely on visual details. Output only the prompt."
    )
    user_msg = f"Prompt: {raw}"
    try:
      enhanced = await provider.generate(
        system_prompt=system_prompt,
        user_message=user_msg,
        temperature=0.7,
        max_tokens=150,
      )
      if enhanced and len(enhanced.strip()) > 10:
        return enhanced.strip()
    except Exception as exc:
      logger.warning("Prompt enhancement skipped due to provider error: %s", exc)

  # Fallback ChatGPT DALL-E 3 Style Rule-Based Prompt Enricher
  boosters = (
    "ChatGPT DALL-E 3 quality masterpiece, photorealistic, 8k uhd, 35mm f/1.8 lens, "
    "cinematic lighting, ray tracing, global illumination, physically based rendering, "
    "studio pristine, hyperrealistic textures, sharp focus, vibrant colors, zero grain"
  )
  if "dall-e" not in raw.lower() and "8k" not in raw.lower():
    return f"{raw}, {boosters}"
  return raw


async def generate_image(
  provider: Optional[ModelProvider] = None,
  *,
  prompt: str,
  style: Optional[str] = "photorealistic",
  width: int = 1024,
  height: int = 1024,
  seed: Optional[int] = None,
  negative_prompt: Optional[str] = None,
  guidance_scale: float = 7.5,
  enhance_prompt_with_ai: bool = True,
  format: str = "PNG",
) -> Dict[str, Any]:
  """Generate image via custom engine with optional LLM prompt enrichment."""
  style = style or "photorealistic"
  t0 = time.perf_counter()

  raw_prompt = prompt.strip()
  if not raw_prompt:
    raise ValueError("Prompt cannot be empty")

  # Dimensions bounds check (supports up to 4K 4096 and 8K 8192)
  width = max(128, min(8192, width))
  height = max(128, min(8192, height))

  # Enhanced prompt via LLM if requested
  final_prompt = raw_prompt
  if enhance_prompt_with_ai and provider:
    final_prompt = await enhance_prompt(provider, raw_prompt, style)

  # Synthesize Image using Custom Model Engine
  img, meta = generate_image_matrix(
    prompt=final_prompt,
    style=style,
    width=width,
    height=height,
    seed=seed,
    negative_prompt=negative_prompt,
    guidance_scale=guidance_scale,
  )

  # Encode Image to Base64
  b64_image = image_to_base64(img, format=format)
  elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

  return {
    "image_b64": b64_image,
    "format": format.upper(),
    "prompt": raw_prompt,
    "enhanced_prompt": final_prompt,
    "style": meta["style"],
    "seed": meta["seed"],
    "width": width,
    "height": height,
    "server_width": meta.get("server_width", img.width),
    "server_height": meta.get("server_height", img.height),
    "negative_prompt": meta["negative_prompt"],
    "guidance_scale": guidance_scale,
    "engine": meta["engine"],
    "elapsed_ms": elapsed_ms,
  }


def get_styles_catalog() -> Dict[str, Any]:
  """Return catalog of available styles."""
  styles = get_available_styles()
  return {
    "styles": styles,
    "count": len(styles),
    "supported_formats": ["PNG", "JPEG"],
    "max_resolution": "2048x2048",
  }
