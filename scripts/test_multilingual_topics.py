"""Test script to verify multi-language topic detection and full response generation."""

import sys
import io
import asyncio
import json

# Ensure UTF-8 output encoding for Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.services import seo_content
from app.engine.seo_content_domains import detect_language


async def test_multilingual():
    test_cases = [
        {"topic": "कॉफी कैसे बनाएं", "expected_lang": "hi", "desc": "Hindi Topic"},
        {"topic": "મોબાઈલ રીપેરિંગ શીખો", "expected_lang": "gu", "desc": "Gujarati Topic"},
        {"topic": "Cómo hacer café en casa", "expected_lang": "es", "desc": "Spanish Topic"},
        {"topic": "Comment apprendre le langage Python", "expected_lang": "fr", "desc": "French Topic"},
        {"topic": "Auto mieten in Deutschland Tipps", "expected_lang": "de", "desc": "German Topic"},
        {"topic": "पुणे मध्ये नवीन व्यवसाय कसा सुरू करावा", "expected_lang": "mr", "desc": "Marathi Topic"},
    ]

    print("=== Testing Multi-Language Topic Generation ===")
    all_passed = True

    for tc in test_cases:
        topic = tc["topic"]
        expected = tc["expected_lang"]
        desc = tc["desc"]

        detected = detect_language(topic)
        print(f"\n--- {desc} ---")
        print(f"Topic: {topic}")
        print(f"Detected Lang: {detected} (expected: {expected})")

        res = await seo_content.generate(
            provider=None,
            topic=topic,
            word_count=300,
            use_ai=False,
            use_rag=False
        )

        lang_result = res.get("language")
        title = res.get("title")
        meta = res.get("meta_description")
        outline = res.get("outline_text", [])
        article = res.get("article", "")
        faqs = res.get("faqs", [])

        print(f"Response Language: {lang_result}")
        print(f"Title: {title}")
        print(f"Meta Description: {meta}")
        print(f"Outline Headings: {outline[:3]}")
        print(f"FAQs Count: {len(faqs)}")
        if faqs:
            print(f"First FAQ Question: {faqs[0].get('question')}")

        if lang_result != expected:
            print(f"❌ FAIL: Expected language {expected}, got {lang_result}")
            all_passed = False
        elif not title or not meta or not article or not faqs:
            print("❌ FAIL: Missing output fields")
            all_passed = False
        else:
            print("✅ PASS: Successfully generated full response in target language!")

    if all_passed:
        print("\n🎉 ALL MULTILINGUAL TOPIC TESTS PASSED SUCCESSFULLY!")
    else:
        print("\n⚠️ SOME TESTS FAILED.")

if __name__ == "__main__":
    asyncio.run(test_multilingual())
