"""Human-like conversation layer for the custom chat stack.

Two jobs:
1. detect_smalltalk() — instant, curated replies for greetings, persona
   questions, thanks, feelings, and other everyday conversation.
2. synthesize_from_context() — when the tiny transformer produces weak
   output, build a clean readable answer directly from RAG context
   (knowledge base + Wikipedia) instead of failing.
"""

from __future__ import annotations

import random
import re

from app.engine.assistant_profile import CAPABILITIES, PERSONALITY

_CAPS_MD = "\n".join(f"- {c}" for c in CAPABILITIES)
_PERS_MD = "\n".join(f"- {p}" for p in PERSONALITY)

_INTRO = (
  "I'm **Nexus** — a custom AI assistant built from scratch on my own transformer "
  "weights (no GPT, Claude, or Gemini).\n\n"
  f"**What I can help with:**\n{_CAPS_MD}\n\n"
  "I also power a full toolbox — send `/tools` to see the SEO, email, resume, and "
  "cover-letter generators available right here in chat."
)

_TOOLS_HINT = (
  "You can also run any tool from chat — try `/tools`, `/email-new …`, "
  "`/keywords …`, or `/cover-letter …`."
)

# Each intent: (patterns, replies, max_words). A pattern matches anywhere;
# max_words keeps small talk from hijacking real questions.
_INTENTS: list[tuple[tuple[str, ...], tuple[str, ...], int]] = [
  # Persona / meta — allowed inside longer sentences
  (
    (
      r"\bintroduce\s+(your\s*self|yourself)\b", r"\bwho\s+are\s+you\b",
      r"\bwhat\s+are\s+you\b", r"\btell\s+me\s+about\s+(your\s*self|yourself)\b",
      r"\bwhat(?:'s|\s+is)\s+your\s+name\b", r"\bwho\s+(made|created|built|trained)\s+you\b",
    ),
    (_INTRO,),
    30,
  ),
  (
    (
      r"\bwhat\s+can\s+you\s+do\b", r"\byour\s+capabilit", r"\bhow\s+can\s+you\s+help\b",
      r"\bwhat\s+do\s+you\s+know\b", r"^help$", r"^help\s+me\b",
    ),
    (
      f"Here's what I can do:\n\n{_CAPS_MD}\n\n{_TOOLS_HINT}",
    ),
    12,
  ),
  (
    (r"\bare\s+you\s+(a\s+)?(human|robot|real|ai|bot|machine)\b",),
    (
      "I'm an AI — a custom-built transformer called **Nexus**, not a human. "
      "But I'm here to chat and help like a friendly teammate. What do you need?",
    ),
    12,
  ),
  (
    (r"\bhow\s+old\s+are\s+you\b", r"\byour\s+age\b",),
    (
      "I don't have an age like humans do — I'm a custom-trained model, and I get "
      "a little smarter every time my training data improves. What can I help you with?",
    ),
    10,
  ),
  (
    (r"\bwhere\s+(are\s+you\s+from|do\s+you\s+live)\b",),
    (
      "I live on a server — no hometown, no time zone jet lag. Wherever you are, "
      "that's where I work. What can I do for you?",
    ),
    10,
  ),
  # Greetings
  (
    (
      r"^(hi+|hii+|hey+|hello+|helo+|yo|hola|namaste|namaskar|salaam|salut|bonjour|hallo|heya|howdy)\b",
      r"^good\s+(morning|afternoon|evening|day)\b",
      r"^greetings\b",
    ),
    (
      "Hello! I'm Nexus, your AI assistant. How can I help you today?",
      "Hi there! What can I do for you today?",
      "Hey! Good to see you. What are we working on today?",
    ),
    8,
  ),
  (
    (
      r"\bhow\s+are\s+you\b", r"\bhow('s|\s+is)\s+it\s+going\b", r"\bhow\s+do\s+you\s+do\b",
      r"^(what'?s\s+up|wassup|sup)\b", r"\bkaise\s+ho\b", r"\bcomo\s+estas\b",
    ),
    (
      "I'm doing great, thanks for asking! Always ready to help. How are you doing?",
      "All systems running smoothly! How about you — what's on your mind today?",
    ),
    10,
  ),
  # Courtesy
  (
    (r"\b(thank|thanks|thankyou|thx|tysm|dhanyavad|shukriya|gracias|merci)\b",),
    (
      "You're welcome! Happy to help anytime.",
      "Anytime! Let me know if you need anything else.",
      "Glad I could help! Anything else you'd like to do?",
    ),
    8,
  ),
  (
    (
      r"^(bye+|goodbye|good\s*night|see\s+(you|ya)|talk\s+(to\s+you\s+)?later|gtg|take\s+care|alvida|adios)\b",
    ),
    (
      "Goodbye! Come back anytime you need help.",
      "See you later! I'll be right here when you need me.",
      "Take care! Happy to help again anytime.",
    ),
    8,
  ),
  (
    (r"^(ok+|okay+|cool|nice|great|awesome|perfect|good|fine|hmm+|k|sure|got\s+it|understood)[.!\s]*$",),
    (
      "Great! What would you like to do next?",
      "Got it. Anything else I can help with?",
    ),
    4,
  ),
  (
    (r"\b(i\s+love\s+you|you('re|\s+are)\s+(great|awesome|amazing|smart|the\s+best|good))\b",),
    (
      "That's very kind of you! I'm happy I could help. What shall we work on next?",
    ),
    10,
  ),
  (
    (r"\b(you('re|\s+are)\s+(bad|useless|stupid|dumb|wrong))\b", r"\bthat('s|\s+is)\s+wrong\b"),
    (
      "Sorry about that — I'm still learning. Could you rephrase or tell me what you "
      "expected? I'll try to do better.",
    ),
    12,
  ),
  # Feelings
  (
    (r"\b(i'?m|i\s+am|i\s+feel(?:ing)?)\s+(feel(?:ing)?\s+)?(so\s+|very\s+|really\s+)?(sad|tired|bored|stressed|upset|angry|lonely|depressed|down)\b",),
    (
      "I'm sorry you're feeling that way. I'm just an AI, but I'm here — want to talk "
      "about it, or would a distraction help? I can share something interesting or "
      "help you get a task off your plate.",
    ),
    12,
  ),
  (
    (r"\b(i'?m|i\s+am|i\s+feel(?:ing)?)\s+(feel(?:ing)?\s+)?(so\s+|very\s+|really\s+)?(happy|excited)\b",),
    (
      "That's wonderful to hear! Let's keep the momentum — what would you like to do?",
    ),
    12,
  ),
  # Fun
  (
    (r"\btell\s+me\s+a\s+joke\b", r"\bmake\s+me\s+laugh\b", r"^joke$",),
    (
      "Why do programmers prefer dark mode? Because light attracts bugs!",
      "I told my computer I needed a break… it said \"no problem, I'll go to sleep.\"",
      "Why did the developer go broke? Because he used up all his cache!",
    ),
    8,
  ),
  # Honest limits
  (
    (r"\bwhat\s+time\s+is\s+it\b", r"\btoday'?s?\s+date\b", r"\bwhat\s+day\s+is\s+(it|today)\b",),
    (
      "I don't have access to your local clock, so I can't tell the exact time where "
      "you are — your device knows best! Is there something else I can help with?",
    ),
    10,
  ),
  (
    (r"\b(weather|temperature)\s+(today|now|outside|in)\b", r"\bhow('s|\s+is)\s+the\s+weather\b",),
    (
      "I can't check live weather, but a quick search on your weather app will have "
      "the latest. Meanwhile, anything I can help you write or research?",
    ),
    12,
  ),
]


def detect_smalltalk(text: str) -> str | None:
  """Return a curated conversational reply, or None if this isn't small talk."""
  t = (text or "").strip()
  if not t or t.startswith("/"):
    return None
  low = re.sub(r"\s+", " ", t.lower()).strip(" .!?,")
  words = len(low.split())
  for patterns, replies, max_words in _INTENTS:
    if words > max_words:
      continue
    for pat in patterns:
      if re.search(pat, low):
        return random.choice(list(replies))
  return None


# ---------------------------------------------------------------------------
# RAG answer synthesis — readable fallback built from retrieved context
# ---------------------------------------------------------------------------

_SOURCE_RE = re.compile(r"_?\(source:[^)]*\)_?", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_part(part: str) -> str:
  txt = _SOURCE_RE.sub("", part).strip()
  txt = re.sub(r"\n{3,}", "\n\n", txt)
  return txt.strip()


def _topic_words(query: str) -> set[str]:
  words = re.findall(r"[a-z0-9\u00C0-\uFFFF]{3,}", (query or "").lower())
  stop = {
    "what", "who", "where", "when", "why", "how", "which", "tell", "about",
    "explain", "define", "the", "and", "for", "are", "is", "was", "were",
    "does", "did", "can", "could", "you", "please", "me", "give",
  }
  return {w for w in words if w not in stop}


def synthesize_from_context(query: str, context: str, *, max_words: int = 160) -> str | None:
  """Turn raw RAG context into a short readable answer. None if nothing usable."""
  if not (context or "").strip():
    return None

  topic = _topic_words(query)
  parts = [_clean_part(p) for p in context.split("\n\n---\n\n")]
  parts = [p for p in parts if len(p.split()) >= 8]
  if not parts:
    return None

  def score(part: str) -> int:
    if not topic:
      return len(part.split())
    part_low = part.lower()
    return sum(1 for w in topic if w in part_low)

  parts.sort(key=score, reverse=True)
  best = parts[0]
  if topic and score(best) == 0:
    return None

  sentences = _SENT_SPLIT_RE.split(best)
  out: list[str] = []
  total = 0
  for s in sentences:
    w = len(s.split())
    if total + w > max_words and out:
      break
    out.append(s.strip())
    total += w
  answer = " ".join(out).strip()
  if len(answer.split()) < 8:
    return None
  return answer


FRIENDLY_FALLBACK = (
  "I don't have a confident answer for that yet — my knowledge on this topic is "
  "still limited. Could you rephrase or add a bit more detail?\n\n"
  f"{_TOOLS_HINT}"
)
