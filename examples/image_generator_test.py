"""Test script for the custom Image Generator model & service engine."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine.image_generator_engine import generate_image_matrix, get_available_styles, image_to_base64
from app.services.image_generator import generate_image, get_styles_catalog


async def test_image_generator():
  print("=== 1. Testing Available Styles Catalog ===")
  catalog = get_styles_catalog()
  print(f"Supported styles count: {catalog['count']}")
  for style in catalog["styles"]:
    print(f" - Style ID: {style['id']:<15} Name: {style['name']}")

  print("\n=== 2. Testing Engine Matrix Generation ===")
  prompt = "Cyberpunk skyline with glowing neon billboards and rain reflection"
  img, meta = generate_image_matrix(
    prompt=prompt,
    style="cyberpunk",
    width=512,
    height=512,
    seed=42,
  )
  print(f"Generated PIL Image size: {img.size}, mode: {img.mode}")
  print(f"Meta details: Engine={meta['engine']}, Seed={meta['seed']}, Style={meta['style']}")
  
  b64 = image_to_base64(img, format="PNG")
  print(f"Base64 string preview: {b64[:60]}... (length: {len(b64)} chars)")

  print("\n=== 3. Testing Service Layer Execution ===")
  result = await generate_image(
    provider=None,
    prompt="Futuristic anime heroine with glowing golden aura",
    style="anime",
    width=512,
    height=512,
    seed=12345,
    enhance_prompt_with_ai=False,
  )
  print(f"Service status: Success!")
  print(f"Format: {result['format']}, Width: {result['width']}, Height: {result['height']}")
  print(f"Elapsed time: {result['elapsed_ms']} ms")
  print("=== All Image Generator tests passed successfully! ===")


if __name__ == "__main__":
  asyncio.run(test_image_generator())
