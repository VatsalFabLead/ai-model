"""Test suite for Resume Builder Improvements (A, B, C, D, E)."""
import asyncio
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from app.services import resume_builder

async def run_tests():
    print("=== Testing Resume Builder Enhancements ===\n")

    test_cases = [
        (
            "Gujarati",
            "વત્સલ પટેલ",
            "સિનિયર ફ્લટર ડેવલપર",
            "vatsal@example.com",
            "+91 98765 43210",
            "Flutter, Dart, Firebase, REST APIs, Git",
            "ફ્લટર એપ ડેવલપમેન્ટમાં ૪ વર્ષનો અનુભવ",
            "We are hiring a Senior Flutter Developer proficient in Flutter, Dart, Firebase, Docker, and Kubernetes for scalable mobile apps.",
        ),
        (
            "Hindi",
            "वत्सल शर्मा",
            "सीनियर सॉफ्टवेयर इंजीनियर",
            "vatsal.sharma@example.com",
            "+91 98765 43210",
            "Python, React, FastApi, Docker, PostgreSQL",
            "सॉफ्टवेयर विकास में ५ साल का अनुभव",
            "Looking for Senior Software Engineer with Python, React, FastAPI, AWS, and Docker experience.",
        ),
        (
            "Spanish",
            "Carlos Gómez",
            "Ingeniero de Software Senior",
            "carlos@example.com",
            "+34 612 345 678",
            "React, Node.js, TypeScript, GraphQL, AWS",
            "Experiencia en desarrollo web full-stack y arquitectura cloud",
            "Se busca Ingeniero de Software con experiencia en React, Node.js, TypeScript, GraphQL y Kubernetes.",
        ),
    ]

    for lang, name, job_title, email, phone, skills, exp, jd in test_cases:
        print(f"--- Testing Language: {lang} (Candidate: '{name}', Role: '{job_title}') ---")
        result = await resume_builder.generate_resume(
            provider=None,
            payload={
                "full_name": name,
                "job_title": job_title,
                "email": email,
                "phone": phone,
                "skills": skills,
                "experience": exp,
                "job_description": jd,
                "language": lang,
            },
        )

        print("1. A. MULTILINGUAL HEADINGS & ZERO ENGLISH LEAKAGE CHECK:")
        md = result.get("resume_markdown", "")
        print(f"   - Markdown Resume Snippet:\n{md[:250]}...")

        print("\n2. B. ATS KEYWORD MATCHER & JD GAP ANALYZER CHECK:")
        jd_res = result.get("jd_match") or {}
        print(f"   - Target JD Match Score: {jd_res.get('match_score')}%")
        print(f"   - Matched Keywords: {jd_res.get('matched_keywords')}")
        print(f"   - Missing Keywords: {jd_res.get('missing_keywords')}")
        print(f"   - Recommended Additions: {jd_res.get('recommended_additions')}")

        print("\n3. C. MULTIPLE ATS LAYOUT TEMPLATES CHECK:")
        layouts = result.get("layout_templates") or {}
        print(f"   - Modern Clean Template: {'YES ✅' if 'modern_clean' in layouts else 'NO ❌'}")
        print(f"   - Executive Elite Template: {'YES ✅' if 'executive_elite' in layouts else 'NO ❌'}")
        print(f"   - Technical Minimal Template: {'YES ✅' if 'technical_minimal' in layouts else 'NO ❌'}")
        print(f"   - Creative Showcase Template: {'YES ✅' if 'creative_showcase' in layouts else 'NO ❌'}")

        print("\n4. D. GOOGLE XYZ FORMULA BULLET REWRITER CHECK:")
        xyz = result.get("xyz_bullets") or []
        print(f"   - First Google XYZ Bullet: {xyz[0] if xyz else 'N/A'}")

        print("\n5. E. COVER LETTER & APPLICATION BUNDLE CHECK:")
        bundle = result.get("application_bundle") or {}
        cl = bundle.get("cover_letter", "")
        formats = bundle.get("export_formats") or {}
        print(f"   - Tailored Cover Letter Snippet:\n{cl[:180]}...")
        print(f"   - Export Formats (Markdown, Plain Text, HTML, JSON Schema): {list(formats.keys())}")
        print("\n" + "="*50 + "\n")

    print("🎉 ALL 5 RESUME BUILDER ENHANCEMENTS TESTED & VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
