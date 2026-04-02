"""
Intent Classifier — Story 8 core
Classifies user messages into learning intents and detects depth level.

Set USE_LOCAL_CLASSIFIER=true to run fully offline with no API key.
Useful for testing routing logic before you have OpenAI credits.
"""

import json
import os
import re
from openai import AsyncOpenAI

USE_LOCAL = os.environ.get("USE_LOCAL_CLASSIFIER", "false").lower() == "true"

#client = AsyncOpenAI() if not USE_LOCAL else None
client = None
if not USE_LOCAL:
    client = AsyncOpenAI()

INTENTS = ["define", "explain", "explore", "challenge", "clarify", "summarize"]

CHALLENGE_SUBTYPES = ["pull", "similar", "simpler", "harder", "general"]

CLASSIFICATION_PROMPT = """You are an intent classifier for an adaptive learning chatbot.
Classify the user's message into exactly one intent and one depth level.

INTENTS:
- define: asking what something is ("what is X?", "define X")
- explain: asking how something works ("how does X work?", "why does X happen?")
- explore: open-ended curiosity ("tell me more", "what else is interesting about X")
- challenge: wants a problem, quiz, or test ("quiz me", "give me a problem", "test me")
- clarify: expressing confusion ("I don't get X", "wait why does X", "that doesn't make sense")
- summarize: wants key points distilled ("summarize", "what are the main points", "tldr")

DEPTH LEVELS (infer from vocabulary, sentence complexity, domain terminology used):
- 1: beginner — simple language, no domain terms, short sentences
- 2: intermediate — some domain familiarity, moderate complexity
- 3: advanced — uses precise domain terminology, complex phrasing, assumes background

CHALLENGE SUBTYPES (only if intent is challenge):
- pull: wants a real textbook problem on a topic ("give me an eigenvalue problem")
- similar: wants a problem like the last one ("give me one like that", "another similar one")
- simpler: wants easier version ("too hard", "simpler version", "easier one")
- harder: wants harder version ("make it harder", "harder version", "more challenging")
- general: generic quiz/challenge request without specifics

MANUAL OVERRIDES (explicit user signals that override inferred depth):
- "explain it simply" / "like I'm a beginner" → depth 1
- "go deeper" / "more technical" / "assume I know the basics" → depth 3

Respond ONLY with valid JSON, no explanation, no markdown:
{
  "intent": "<intent>",
  "depth": <1|2|3>,
  "challenge_subtype": "<subtype or null>",
  "topic": "<main topic being asked about, brief>",
  "manual_override": <true|false>
}"""


def _local_classify(message: str) -> dict:
    """
    Rule-based classifier — no API key needed.
    Good enough for testing routing logic locally.
    Enable with: export USE_LOCAL_CLASSIFIER=true
    """
    m = message.lower()

    # ── Intent ──
    if re.search(r'\b(what is|define|definition of|meaning of)\b', m):
        intent, subtype = "define", None
    elif re.search(r'\b(i don\'?t (get|understand)|confused|doesn\'?t make sense|wait why|what do you mean)\b', m):
        intent, subtype = "clarify", None
    elif re.search(r'\b(summarize|summary|key points|main points|tldr|overview)\b', m):
        intent, subtype = "summarize", None
    elif re.search(r'\b(tell me more|more about|what else|explore|interesting about)\b', m):
        intent, subtype = "explore", None
    elif re.search(r'\b(quiz|test me|give me a problem|textbook problem|practice problem|question on|challenge me)\b', m):
        intent = "challenge"
        if re.search(r'textbook|give me a .{0,20}problem|pull', m):
            subtype = "pull"
        elif re.search(r'similar|like that|another one', m):
            subtype = "similar"
        elif re.search(r'simpler|easier|too hard', m):
            subtype = "simpler"
        elif re.search(r'harder|more challenging|level up', m):
            subtype = "harder"
        else:
            subtype = "general"
    elif re.search(r'\b(harder|more challenging)\b', m):
        intent, subtype = "challenge", "harder"
    elif re.search(r'\b(simpler|easier)\b', m):
        intent, subtype = "challenge", "simpler"
    elif re.search(r'\b(similar|like that)\b', m):
        intent, subtype = "challenge", "similar"
    elif re.search(r'\b(how does|how do|why does|why do|explain|walk me through|how is|how are)\b', m):
        intent, subtype = "explain", None
    else:
        intent, subtype = "explain", None

    # ── Depth ──
    manual_override = False
    if re.search(r'simply|like i\'?m a (beginner|kid)|eli5|no jargon|basic', m):
        depth, manual_override = 1, True
    elif re.search(r'go deeper|more technical|assume i know|advanced|rigorous|in depth', m):
        depth, manual_override = 3, True
    else:
        domain_terms = r'eigenvalue|fourier|hamiltonian|manifold|entropy|thermodynamic|diagonaliz|wave.particle|quantum|relativit|lagrangian|gradient|divergence|curl|topology|differential equation'
        word_count = len(message.split())
        has_domain = bool(re.search(domain_terms, m))
        if has_domain and word_count > 15:
            depth = 3
        elif has_domain or word_count > 10:
            depth = 2
        else:
            depth = 1

    # ── Topic ──
    stop = {'what','is','a','an','the','how','does','do','why','give','me','tell','about','explain','please','can','you','i','my','on','of','for'}
    words = [w for w in re.sub(r'[^\w\s]', '', m).split() if w not in stop and len(w) > 2]
    topic = " ".join(words[:4]) if words else message[:40]

    return {
    "intent": intent,
    "depth": depth,
    "depth_label": ["", "beginner", "intermediate", "advanced"][depth],
    "challenge_subtype": subtype,
    "topic": topic,
    "manual_override": manual_override
}


async def classify_intent(message: str, conversation_history: list = None) -> dict:
    """
    Classify a user message into intent + depth + challenge subtype.
    Returns structured dict ready for prompt router.
    """
    if USE_LOCAL:
        return _local_classify(message)

    context = ""
    if conversation_history:
        recent = conversation_history[-3:]
        context = "\nRecent conversation:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
        )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": f"Message: {message}{context}"}
        ],
        temperature=0.1,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "intent": "explain",
                "depth": 2,
                "challenge_subtype": None,
                "topic": message[:50],
                "manual_override": False
            }

    result["intent"] = result.get("intent", "explain")
    if result["intent"] not in INTENTS:
        result["intent"] = "explain"

    result["depth"] = max(1, min(3, int(result.get("depth", 2))))

    if result["intent"] == "challenge":
        subtype = result.get("challenge_subtype", "general")
        if subtype not in CHALLENGE_SUBTYPES:
            result["challenge_subtype"] = "general"
    else:
        result["challenge_subtype"] = None

    return result


def apply_session_depth(classified: dict, session_depth_history: list) -> dict:
    """
    Adjust depth based on session trajectory if no manual override.
    If user has been asking for simpler/clarify repeatedly, nudge depth down.
    If user is using advanced terms consistently, nudge up.
    """
    if classified.get("manual_override") or len(session_depth_history) < 3:
        return classified

    recent_depths = session_depth_history[-5:]
    avg = sum(recent_depths) / len(recent_depths)

    current = classified["depth"]
    if avg < current - 0.7:
        classified["depth"] = max(1, current - 1)
    elif avg > current + 0.7:
        classified["depth"] = min(3, current + 1)

    return classified