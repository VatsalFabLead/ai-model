"""Topic-domain content packs — real articles, not generic SEO boilerplate.

Detects fitness, health, tech, food, etc. and generates varied, topic-relevant copy.
100% custom templates — no GPT/Claude/Gemini.
"""

from __future__ import annotations

import random
import re
import secrets
import time
from typing import Any

_DOMAIN_SIGNALS: dict[str, list[str]] = {
  "fitness": [
    "workout", "exercise", "fitness", "gym", "yoga", "muscle", "cardio", "strength",
    "bodyweight", "training", "stretch", "warm-up", "warmup", "beginner workout",
    "home workout", "at-home", "weight loss", "calories", "reps", "sets",
  ],
  "health": [
    "health", "wellness", "nutrition", "diet", "sleep", "mental health", "stress",
    "meditation", "vitamin", "immune", "hydration", "healthy lifestyle",
  ],
  "food": [
    "recipe", "cooking", "meal", "breakfast", "dinner", "baking", "cuisine", "food",
    "ingredient", "kitchen",
  ],
  "tech": [
    "software", "programming", "code", "app", "flutter", "python", "javascript",
    "api", "cloud", "ai", "machine learning", "developer", "web", "mobile",
  ],
  "business": [
    "marketing", "sales", "startup", "ecommerce", "seo", "email marketing",
    "conversion", "branding", "advertising", "lead", "revenue",
  ],
  "travel": [
    "travel", "hotel", "flight", "destination", "tourism", "vacation", "itinerary",
  ],
  "education": [
    "learn", "course", "study", "exam", "student", "tutorial", "lesson", "skill",
  ],
  "enterprise": [
    "erp", "crm", "enterprise resource", "inventory management", "manufacturing erp",
    "supply chain", "accounting software", "hrms", "procurement", "warehouse management",
    "odoo", "sap", "business software", "cloud erp",
  ],
}

_SECONDARY_SUGGESTIONS: dict[str, list[str]] = {
  "fitness": [
    "beginner workout plan", "home exercise routine", "bodyweight exercises",
    "at-home fitness", "beginner fitness routine", "daily workout plan",
    "strength training at home", "no equipment workout", "healthy lifestyle",
  ],
  "health": [
    "wellness tips", "healthy habits", "nutrition guide", "self-care routine",
    "mental wellness", "daily health routine",
  ],
  "food": [
    "easy recipes", "meal prep", "cooking tips", "ingredient list", "healthy meals",
  ],
  "tech": [
    "best practices", "step-by-step guide", "tutorial", "tips and tricks", "setup guide",
  ],
  "business": [
    "best practices", "strategy guide", "tips for beginners", "growth tactics",
  ],
  "travel": [
    "travel tips", "packing list", "budget travel", "things to do", "local guide",
  ],
  "education": [
    "study tips", "learning path", "beginner guide", "practice exercises",
  ],
  "enterprise": [
    "ERP modules", "implementation guide", "vendor comparison", "ROI calculator",
    "inventory tracking", "financial reporting", "manufacturing workflow", "cloud deployment",
    "data migration", "user training",
  ],
  "general": [
    "beginner guide", "step-by-step", "best practices", "tips and tricks",
    "how to get started", "common mistakes",
  ],
}


def make_variation_seed(explicit: int | None = None) -> int:
  if explicit is not None:
    return int(explicit) & 0x7FFFFFFF
  return secrets.randbits(31)


def _pick(seed: int, options: list[str]) -> str:
  if not options:
    return ""
  return options[seed % len(options)]


def detect_domain(topic: str, keywords: list[str]) -> str:
  text = f"{topic} {' '.join(keywords)}".lower()
  scores = {d: sum(1 for s in sigs if s in text) for d, sigs in _DOMAIN_SIGNALS.items()}
  best = max(scores, key=scores.get)
  return best if scores[best] > 0 else "general"


def expand_keywords(topic: str, keywords: list[str], domain: str) -> dict[str, Any]:
  primary = (keywords[0] if keywords else topic).strip()
  secondary: list[str] = []
  seen = {primary.lower()}
  for kw in keywords[1:]:
    k = kw.strip()
    if k and k.lower() not in seen:
      secondary.append(k)
      seen.add(k.lower())
  pool = _SECONDARY_SUGGESTIONS.get(domain, _SECONDARY_SUGGESTIONS["general"])
  for s in pool:
    if s.lower() not in seen and len(secondary) < 9:
      secondary.append(s)
      seen.add(s.lower())
  return {"primary": primary, "secondary": secondary}


def build_structured_outline(
  topic: str,
  primary: str,
  *,
  domain: str,
  category: str,
  seed: int,
) -> list[dict[str, str]]:
  h1 = _pick(seed, [
    f"{topic}: A Professional Guide",
    f"{topic}: Complete Beginner Guide",
    f"{primary.title()}: Expert Guide for Beginners",
  ])
  if domain == "fitness":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"Benefits of {primary.title()} for Beginners"},
      {"level": "h2", "text": "Essential Tips Before Starting"},
      {"level": "h2", "text": f"Beginner {primary.title()} Plan"},
      {"level": "h3", "text": "Warm-Up Exercises"},
      {"level": "h3", "text": "Upper Body Exercises"},
      {"level": "h3", "text": "Lower Body Exercises"},
      {"level": "h3", "text": "Core Exercises"},
      {"level": "h3", "text": "Cool-Down and Stretching"},
      {"level": "h2", "text": "Weekly Workout Schedule"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "How to Stay Consistent"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if domain == "enterprise":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": f"What Is {primary.title()}?"},
      {"level": "h2", "text": "Core Modules and Features"},
      {"level": "h3", "text": "Finance and Accounting"},
      {"level": "h3", "text": "Inventory and Supply Chain"},
      {"level": "h3", "text": "Manufacturing and Operations"},
      {"level": "h3", "text": "HR and Payroll"},
      {"level": "h2", "text": f"Benefits of {primary.title()} for Businesses"},
      {"level": "h2", "text": "How to Choose the Right Solution"},
      {"level": "h2", "text": "Implementation Roadmap"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "Conclusion"},
    ]
  if category == "how_to_guide":
    return [
      {"level": "h1", "text": h1},
      {"level": "h2", "text": "Introduction"},
      {"level": "h2", "text": "What You Need Before Starting"},
      {"level": "h2", "text": "Step-by-Step Instructions"},
      {"level": "h3", "text": "Step 1: Prepare"},
      {"level": "h3", "text": "Step 2: Execute"},
      {"level": "h3", "text": "Step 3: Refine"},
      {"level": "h2", "text": "Common Mistakes to Avoid"},
      {"level": "h2", "text": "Conclusion"},
    ]
  return [
    {"level": "h1", "text": h1},
    {"level": "h2", "text": "Introduction"},
    {"level": "h2", "text": f"Why {primary.title()} Matters"},
    {"level": "h2", "text": f"Key Benefits of {primary.title()}"},
    {"level": "h2", "text": "Practical Tips and Best Practices"},
    {"level": "h2", "text": f"How to Get Started With {primary.title()}"},
    {"level": "h2", "text": "Common Mistakes to Avoid"},
    {"level": "h2", "text": "Conclusion"},
  ]


def _fitness_article(topic: str, primary: str, tone: str, seed: int) -> tuple[str, str, str]:
  title = _pick(seed, [
    f"{topic}: A Professional Beginner Workout Plan for Better Fitness",
    f"{primary.title()} for Beginners: Complete Home Fitness Guide",
    f"{topic} — Step-by-Step Plan for Strength and Endurance",
  ])
  meta = _trim(
    f"Discover an effective {primary} with a step-by-step beginner workout plan. "
    "Learn essential exercises, weekly schedules, and tips to build strength "
    "and improve overall fitness from home."
  )
  intro = _pick(seed + 1, [
    (
      f"Starting a fitness journey can seem challenging, especially for beginners. "
      f"However, a structured **{primary}** routine provides a convenient and affordable way "
      "to improve overall health without a gym membership. With consistency and the right exercises, "
      "beginners can develop strength, flexibility, and endurance from home."
    ),
    (
      f"A **{primary}** is one of the most accessible ways to begin improving your health. "
      "No expensive equipment is required — just space, commitment, and a clear plan. "
      "This guide walks you through everything a beginner needs to train safely and effectively."
    ),
  ])
  body = f"""# {title}

## Introduction

{intro}

## Benefits of {primary.title()} for Beginners

Home workouts offer flexibility and convenience. They eliminate travel time, require little to no equipment, and fit any schedule. Regular activity improves cardiovascular health, boosts energy, supports weight management, and reduces stress.

## Essential Tips Before Starting

Before beginning, remember to:

- Set realistic fitness goals
- Start slowly and focus on proper form
- Stay hydrated throughout the day
- Wear comfortable clothing and supportive footwear
- Always include warm-up and cool-down sessions

## Beginner {primary.title()} Plan

### Warm-Up Exercises

Begin with five minutes of light cardio: marching in place, arm circles, and jumping jacks prepare your body for exercise.

### Upper Body Exercises

- **Push-ups:** 3 sets of 10 repetitions
- **Incline push-ups:** 3 sets of 12 repetitions

### Lower Body Exercises

- **Squats:** 3 sets of 15 repetitions
- **Lunges:** 3 sets of 12 repetitions per leg

### Core Exercises

- **Plank:** Hold 30 seconds, 3 rounds
- **Bicycle crunches:** 3 sets of 20 repetitions

### Cool-Down and Stretching

Spend five to ten minutes stretching major muscle groups to improve flexibility and reduce soreness.

## Weekly Workout Schedule

| Day | Activity |
|-----|----------|
| Monday | Full body workout |
| Tuesday | Cardio and stretching |
| Wednesday | Upper body training |
| Thursday | Rest or walking |
| Friday | Lower body training |
| Saturday | Core exercises |
| Sunday | Recovery and stretching |

## Common Mistakes to Avoid

Many beginners overtrain or expect immediate results. Skipping warm-ups, ignoring proper technique, or lacking consistency slows progress. Focus on gradual improvement rather than perfection.

## How to Stay Consistent

Track progress, follow your weekly schedule, and maintain a balanced diet. Even 30 minutes of daily exercise leads to noticeable improvements over time.

## Conclusion

A well-designed **{primary}** routine is an excellent starting point for beginners. With discipline, patience, and gradual progress, anyone can build strength and healthier habits without leaving home.
"""
  return title, meta, body.strip()


def _enterprise_article(
  topic: str,
  primary: str,
  tone: str,
  seed: int,
  audience: str | None,
) -> tuple[str, str, str]:
  aud = f" for {audience}" if audience else ""
  title = _pick(seed, [
    f"{topic}: Complete Guide to Selection and Implementation{aud}",
    f"{primary.title()} — Features, Benefits, and Best Practices",
    f"How to Choose and Deploy {primary.title()} Successfully",
  ])
  meta = _trim(
    f"Learn what {primary} is, core modules, selection criteria, and implementation steps. "
    f"Practical guide for manufacturing, inventory, and finance teams{aud}."
  )
  intro = _pick(seed + 1, [
    (
      f"**{primary.title()}** integrates finance, inventory, manufacturing, HR, and reporting "
      "into one system so teams stop juggling spreadsheets and disconnected tools."
    ),
    (
      f"Organizations adopt **{primary}** to unify operations, improve data accuracy, "
      "and speed up decisions across departments."
    ),
  ])
  body = f"""# {title}

## Introduction

{intro}

## What Is {primary.title()}?

Enterprise Resource Planning (ERP) software connects core business processes in a single platform. Instead of isolated apps for accounting, stock, and production, ERP gives one shared database and workflow.

## Core Modules and Features

### Finance and Accounting

General ledger, accounts payable/receivable, tax compliance, and real-time financial reporting.

### Inventory and Supply Chain

Track stock levels, purchase orders, suppliers, and warehouse movements with accurate reorder alerts.

### Manufacturing and Operations

Plan production, bill of materials (BOM), work orders, and shop-floor scheduling for make-to-stock or make-to-order models.

### HR and Payroll

Employee records, attendance, payroll runs, and basic workforce analytics in the same system.

## Benefits of {primary.title()} for Businesses

- **Single source of truth** — one dataset for sales, inventory, and finance
- **Fewer manual errors** — automated postings between modules
- **Faster reporting** — dashboards for cash flow, margins, and stock aging
- **Scalable growth** — add users, warehouses, or legal entities without rebuilding processes

## How to Choose the Right Solution

1. Map your must-have modules (finance, inventory, MRP, CRM, etc.)
2. Compare cloud vs on-premise and total cost of ownership
3. Check integration with existing tools (e-commerce, POS, BI)
4. Run a pilot with real data from one department
5. Verify vendor support, training, and migration services

## Implementation Roadmap

| Phase | Focus |
|-------|--------|
| Discovery | Process mapping, gap analysis, data audit |
| Design | Workflows, roles, chart of accounts, item masters |
| Build | Configuration, integrations, test environment |
| Test | UAT, parallel run, fix reconciliations |
| Go-live | Cutover, hypercare support, training |

## Common Mistakes to Avoid

- Buying more modules than the team can adopt in year one
- Skipping data cleanup before migration
- Weak change management — users revert to spreadsheets
- No executive sponsor for cross-department decisions

## Conclusion

The right **{primary}** implementation pays off through visibility, automation, and faster closes. Start with clear goals, clean master data, and phased rollout.
"""
  return title, meta, body.strip()


def _general_article(
  topic: str,
  primary: str,
  *,
  domain: str,
  tone: str,
  seed: int,
  audience: str | None,
) -> tuple[str, str, str]:
  aud = f" for {audience}" if audience else ""
  title = _pick(seed, [
    f"{topic}: A Complete Guide{aud}",
    f"{primary.title()}: Practical Guide for Beginners",
    f"Everything You Need to Know About {primary.title()}",
  ])
  meta = _trim(
    f"Learn about {primary} with clear, practical guidance. "
    f"Covers benefits, step-by-step tips, and expert advice{aud}."
  )
  benefit = {
    "health": "supports physical and mental well-being, builds sustainable habits, and improves daily energy",
    "food": "saves time, improves nutrition, and makes home cooking enjoyable for any skill level",
    "tech": "helps you work smarter, avoid common pitfalls, and build reliable solutions faster",
    "business": "drives measurable growth, improves customer trust, and creates long-term value",
    "travel": "helps you plan smarter trips, save money, and enjoy destinations more fully",
    "education": "accelerates learning, builds confidence, and creates lasting skills",
    "general": "saves time, reduces mistakes, and delivers practical results you can apply immediately",
  }.get(domain, "delivers practical, real-world benefits you can apply right away")

  intro_variants = [
    (
      f"Understanding **{primary}** is valuable for anyone who wants clear, actionable guidance. "
      f"This article explains what matters most, how to begin, and which mistakes to avoid."
    ),
    (
      f"Whether you are just starting out or refining your approach, **{primary}** offers real benefits. "
      f"This guide breaks the topic into simple steps anyone can follow."
    ),
  ]
  body = f"""# {title}

## Introduction

{_pick(seed + 2, intro_variants)}

## Why {primary.title()} Matters

{primary.title()} {benefit}. Investing time in the right approach pays off through better outcomes and fewer setbacks.

## Key Benefits of {primary.title()}

- Practical knowledge you can apply immediately
- Clear structure that saves time and reduces confusion
- Confidence to make informed decisions
- Sustainable habits that compound over weeks and months

## Practical Tips and Best Practices

- Start with a clear goal and realistic timeline
- Learn fundamentals before advanced techniques
- Track progress and adjust based on results
- Seek reliable sources and proven methods
- Stay consistent — small daily steps beat occasional bursts

## How to Get Started With {primary.title()}

1. Research the basics and define what success looks like for you
2. Gather any tools or resources you need
3. Follow a structured plan for the first 2–4 weeks
4. Review results and refine your approach

## Common Mistakes to Avoid

- Rushing without understanding fundamentals
- Expecting instant results instead of gradual progress
- Skipping planning and jumping straight to execution
- Giving up too early before habits take hold

## Conclusion

**{primary.title()}** becomes manageable when you follow a clear plan. Apply these steps consistently, track your progress, and refine your approach over time.
"""
  return title, meta, body.strip()


def build_domain_faqs(topic: str, primary: str, domain: str, seed: int) -> list[dict[str, str]]:
  if domain == "fitness":
    variants = [
      [
        {
          "question": f"What is the best {primary} for beginners?",
          "answer": "A combination of squats, push-ups, lunges, planks, and stretching provides a balanced beginner workout.",
        },
        {
          "question": "Can beginners build muscle with home workouts?",
          "answer": "Yes. Bodyweight exercises and progressive overload help beginners build muscle and improve strength.",
        },
        {
          "question": "How long should a beginner exercise at home?",
          "answer": "Aim for 20 to 45 minutes of exercise, three to five days per week.",
        },
        {
          "question": "Do I need equipment for a home workout?",
          "answer": "No. Most beginner exercises can be performed using body weight alone.",
        },
        {
          "question": "How long does it take to see results from home workouts?",
          "answer": "Most people notice improvements in energy and fitness within four to eight weeks of consistent training.",
        },
      ],
      [
        {
          "question": f"How often should I do a {primary}?",
          "answer": "Three to five sessions per week with rest days allows muscles to recover and adapt.",
        },
        {
          "question": "Is warming up necessary for beginners?",
          "answer": "Yes. Five minutes of light movement reduces injury risk and improves performance.",
        },
        {
          "question": f"What exercises are safest for a {primary}?",
          "answer": "Bodyweight squats, incline push-ups, lunges, and planks are safe, effective starter movements.",
        },
        {
          "question": "Can I lose weight with home workouts alone?",
          "answer": "Exercise combined with a balanced diet supports weight management; consistency matters most.",
        },
      ],
    ]
    return variants[seed % len(variants)]

  if domain == "enterprise":
    return [
      {
        "question": f"What is {primary}?",
        "answer": (
          f"{primary.title()} is integrated business software that connects finance, inventory, "
          "manufacturing, HR, and reporting in one platform."
        ),
      },
      {
        "question": f"What modules are included in {primary}?",
        "answer": "Typical modules include accounting, inventory, procurement, manufacturing, CRM, and HR/payroll.",
      },
      {
        "question": f"How long does {primary} implementation take?",
        "answer": "Small businesses may go live in 3–6 months; mid-size deployments often take 6–12 months depending on scope.",
      },
      {
        "question": f"Cloud or on-premise {primary} — which is better?",
        "answer": "Cloud ERP offers faster updates and lower upfront cost; on-premise suits strict data-residency or heavy customization needs.",
      },
      {
        "question": f"Who needs {primary}?",
        "answer": "Growing companies with complex inventory, multi-location operations, or manual reconciliation pain benefit most.",
      },
    ]

  return [
    {
      "question": f"What is {primary}?",
      "answer": f"{primary.title()} refers to practical methods and knowledge related to {topic}, applied step by step for real results.",
    },
    {
      "question": f"How do I get started with {primary}?",
      "answer": "Begin with the fundamentals, set a clear goal, follow a structured plan, and track your progress weekly.",
    },
    {
      "question": f"How long until I see results with {primary}?",
      "answer": "Most people notice meaningful progress within a few weeks when they apply the steps consistently.",
    },
    {
      "question": f"Who benefits most from learning about {primary}?",
      "answer": f"Beginners, enthusiasts, and anyone who wants practical guidance on {topic} without unnecessary complexity.",
    },
  ]


def build_rich_content(
  topic: str,
  keywords: list[str],
  *,
  category: str,
  tone: str,
  audience: str | None,
  seed: int,
) -> dict[str, Any]:
  domain = detect_domain(topic, keywords)
  kw = expand_keywords(topic, keywords, domain)
  primary = kw["primary"]
  outline = build_structured_outline(topic, primary, domain=domain, category=category, seed=seed)

  if domain == "fitness":
    title, meta, article = _fitness_article(topic, primary, tone, seed)
  elif domain == "enterprise":
    title, meta, article = _enterprise_article(topic, primary, tone, seed, audience)
  else:
    title, meta, article = _general_article(
      topic, primary, domain=domain, tone=tone, seed=seed, audience=audience,
    )

  faqs = build_domain_faqs(topic, primary, domain, seed)
  return {
    "metadata": {"title": title, "meta_description": meta},
    "keywords": kw,
    "outline": outline,
    "content": {"article": article, "tone": tone},
    "faqs": faqs,
    "domain": domain,
    "variation_seed": seed,
  }


def _trim(meta: str, limit: int = 160) -> str:
  meta = re.sub(r"\s+", " ", meta.strip())
  if len(meta) <= limit:
    return meta
  return meta[: limit - 3].rsplit(" ", 1)[0].rstrip() + "..."
