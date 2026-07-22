"""Custom Image Generator API — neural & procedural text-to-image synthesis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import get_tool_provider
from app.core.security import verify_api_key
from app.services import image_generator

router = APIRouter(prefix="/image-generator", tags=["image-generator"])


class ImageGenerateRequest(BaseModel):
  prompt: str = Field(..., min_length=1, max_length=1000, examples=["Futuristic city at night with glowing lights"])
  style: Optional[str] = Field(default="photorealistic", description="Artistic style preset")
  width: int = Field(default=1024, ge=128, le=8192, description="Image width in pixels (supports 4K 4096 and 8K 8192)")
  height: int = Field(default=1024, ge=128, le=8192, description="Image height in pixels (supports 4K 4096 and 8K 8192)")
  seed: Optional[int] = Field(default=None, description="Random seed for deterministic generation")
  negative_prompt: Optional[str] = Field(default=None, description="Attributes to avoid in synthesis")
  guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Prompt conditioning weight")
  enhance_prompt: bool = Field(default=True, description="Enrich prompt details using custom LLM engine")
  format: str = Field(default="PNG", description="Output format: PNG or JPEG")


class ImageGenerateResponse(BaseModel):
  image_b64: str = Field(..., description="Base64 data URL string of the generated image")
  format: str
  prompt: str
  enhanced_prompt: str
  style: str
  seed: int
  width: int
  height: int
  server_width: Optional[int] = None
  server_height: Optional[int] = None
  negative_prompt: Optional[str] = None
  guidance_scale: float
  engine: str
  elapsed_ms: float


@router.get("/styles")
async def list_styles(_: str = Depends(verify_api_key)) -> Dict[str, Any]:
  """Retrieve catalog of supported artistic style presets and resolution capabilities."""
  return image_generator.get_styles_catalog()


@router.post("/generate", response_model=ImageGenerateResponse)
async def generate(
  payload: ImageGenerateRequest,
  request: Request,
  _: str = Depends(verify_api_key),
) -> ImageGenerateResponse:
  """Generate image from text prompt using custom neural procedural matrix model."""
  provider = None
  if payload.enhance_prompt:
    provider = get_tool_provider(request)

  try:
    result = await image_generator.generate_image(
      provider,
      prompt=payload.prompt,
      style=payload.style,
      width=payload.width,
      height=payload.height,
      seed=payload.seed,
      negative_prompt=payload.negative_prompt,
      guidance_scale=payload.guidance_scale,
      enhance_prompt_with_ai=payload.enhance_prompt,
      format=payload.format,
    )
  except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc

  return JSONResponse(
    content=ImageGenerateResponse(**result).model_dump(),
    headers={"X-Image-Engine": result["engine"]},
  )
