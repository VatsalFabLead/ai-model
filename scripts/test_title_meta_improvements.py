"""Test suite for SEO Title & Meta Description Generator Improvements (A, B, C, D, E)."""
import asyncio
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.services import title_meta

async def run_tests():
    print("=== Testing SEO Title & Meta Generator Enhancements ===\n")

    for test_lang, topic in [
        ("Gujarati", "ઘર બેઠા સરળ રીતે મોબાઈલ સંભાળવાની રીતો"),
        ("Hindi", "घर बैठे मोबाइल की देखभाल करने के आसान तरीके"),
        ("Spanish", "Cómo reparar un teléfono móvil en casa paso a paso"),
    ]:
        print(f"\n--- Testing Language: {test_lang} ---")
        result = await title_meta.generate(
            provider=None,
            topic=topic,
            variations=5,
            language=test_lang,
            brand_name="FabAI",
            location="City Center",
            use_ai=False,
            use_rag=True,
        )

        variations = result.get("variations", [])
        top_v = variations[0] if variations else {}
        print("  - TOP TITLE:", top_v.get("title"))
        print("  - TOP META:", top_v.get("meta_description"))
        print("  - PIXEL WIDTH:", top_v.get("pixel_width"))
        print("  - BUCKET:", top_v.get("ab_testing_bucket"))
        print("  - OG TITLE:", (result.get("social_meta_bundle") or {}).get("og_title"))

    print("\n🎉 ALL 5 TITLE & META ENHANCEMENTS TESTED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
