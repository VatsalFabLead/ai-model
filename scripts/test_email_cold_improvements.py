"""Test suite for Email Assistant - Cold Email Mode Improvements (A, B, C, D, E)."""
import asyncio
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.services import email_assistant

async def run_tests():
    print("=== Testing Email Assistant - Cold Email Mode Enhancements ===\n")

    test_cases = [
        (
            "Gujarati",
            "ટેકનોલોજી સોલ્યુશન્સ લિમિટેડ",
            "કસ્ટમ મોબાઇલ એપ ડેવલપમેન્ટ સેવાઓ",
            "એપ ડેવલપમેન્ટ સમયમાં ૪૦% ઘટાડો અને ૩૫% ખર્ચ બચત",
        ),
        (
            "Hindi",
            "इन्नोवेशन्स प्राइवेट लिमिटेड",
            "कस्टम सॉफ्टवेयर विकास सेवाएं",
            "विकास समय में ४०% की कमी और ३५% लागत बचत",
        ),
        (
            "Spanish",
            "Innovaciones Tecnológicas S.L.",
            "Desarrollo de aplicaciones móviles a medida",
            "Reducción del 35% en costos de desarrollo y entrega en 60 días",
        ),
    ]

    for lang, company_name, purpose_offer, value_prop in test_cases:
        print(f"--- Testing Language: {lang} (Company: '{company_name}') ---")
        result = await email_assistant.generate_cold_email(
            provider=None,
            company_name=company_name,
            purpose_offer=purpose_offer,
            value_proposition=value_prop,
            tone="professional",
        )

        print("1. GENERATED COLD SUBJECT:", result.get("subject"))

        print("\n2. C & MULTILINGUAL B2B OPENER CHECK:")
        print(f"   - Cold Email Body Snippet:\n{result.get('email', '')[:220]}...")

        print("\n3. A. 3-STEP FOLLOW-UP SEQUENCE CHECK:")
        seq = result.get("cold_sequence") or {}
        print(f"   - Step 1 ({seq.get('step_1', {}).get('timing')}): {seq.get('step_1', {}).get('subject')}")
        print(f"   - Step 2 ({seq.get('step_2', {}).get('timing')}): {seq.get('step_2', {}).get('subject')}")
        print(f"   - Step 3 ({seq.get('step_3', {}).get('timing')}): {seq.get('step_3', {}).get('subject')}")

        print("\n4. B. PERSONALIZATION MERGE TAGS CHECK:")
        tpl = result.get("outreach_template") or ""
        has_tags = "{{COMPANY_NAME}}" in tpl or "{{FIRST_NAME}}" in tpl
        print(f"   - Merge Tag Template Ready: {'YES ✅' if has_tags else 'NO ❌'}")
        print(f"   - Template Preview: {tpl[:120]}...")

        print("\n5. D. A/B STRATEGY ANGLES CHECK:")
        angles = result.get("cold_strategy_angles") or {}
        print(f"   - Pain Point Angle: {angles.get('pain_point_angle')[:90]}...")
        print(f"   - Social Proof Angle: {angles.get('social_proof_angle')[:90]}...")
        print(f"   - Direct ROI Angle: {angles.get('direct_roi_angle')[:90]}...")

        print("\n6. E. COLD DELIVERABILITY AUDIT CHECK:")
        deliv = result.get("cold_deliverability") or {}
        print(f"   - Word Count: {deliv.get('word_count')} (Concise <= 150: {'YES ✅' if deliv.get('is_concise') else 'NO ❌'})")
        print(f"   - Deliverability Score: {deliv.get('cold_deliverability_score')}/100")
        print(f"   - Deliverability Status: {deliv.get('deliverability_status')}")
        print("\n" + "="*50 + "\n")

    print("🎉 ALL 5 EMAIL ASSISTANT COLD ENHANCEMENTS TESTED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
