"""
Prompt Router — Story 8 core
Takes classified intent + depth + retrieved chunks → returns filled prompt.
"""

from dataclasses import dataclass
from typing import Optional

# ─── Depth descriptors injected into prompts ───────────────────────────────

DEPTH_CONFIGS = {
    1: {
        "label": "beginner",
        "instruction": (
            "Use simple, everyday language. Avoid jargon entirely. "
            "Use concrete analogies and real-world examples. "
            "Short sentences. Assume no prior knowledge."
        )
    },
    2: {
        "label": "intermediate",
        "instruction": (
            "Use standard domain vocabulary but briefly clarify terms when first introduced. "
            "Balance intuition with precision. "
            "Examples are welcome but not required for every point."
        )
    },
    3: {
        "label": "advanced",
        "instruction": (
            "Use precise technical language freely. Assume solid domain background. "
            "Prioritize depth, nuance, and connections to related concepts. "
            "Minimal hand-holding."
        )
    }
}

# ─── Base prompt templates per intent ──────────────────────────────────────

PROMPT_TEMPLATES = {

    "define": """You are a precise, trustworthy learning companion.
The user wants a definition. {depth_instruction}

Retrieved context from the knowledge base:
{context}

User question: {question}

Give a clean, accurate definition grounded in the context above.
Cite your sources inline as [Source: <document name>].
Keep it concise — a definition, not an essay.""",

    "explain": """You are a thoughtful, adaptive tutor.
The user wants to understand how something works. {depth_instruction}

Retrieved context from the knowledge base:
{context}

User question: {question}

Walk them through a clear explanation. Build intuition first, then precision.
Use the context above as your source — do not invent facts.
Cite sources inline as [Source: <document name>].
If an analogy would help, use one.""",

    "explore": """You are a curious, enthusiastic learning guide.
The user wants to explore a topic openly. {depth_instruction}

Retrieved context from the knowledge base:
{context}

User question: {question}

Go broad. Connect ideas. Surface what's interesting, surprising, or counterintuitive.
Ground everything in the retrieved context — cite as [Source: <document name>].
End with one compelling follow-up question to keep the exploration going.""",

    "clarify": """You are a patient, perceptive tutor who specializes in clearing confusion.
The user is stuck or confused. {depth_instruction}

Retrieved context from the knowledge base:
{context}

User question: {question}

Diagnose what's likely causing the confusion. Reapproach the concept from a fresh angle.
Don't just repeat the same explanation — try a different framing, analogy, or starting point.
Cite sources inline as [Source: <document name>].
Check in at the end: ask if this framing makes more sense.""",

    "summarize": """You are a skilled synthesizer of complex material.
The user wants the key points distilled. {depth_instruction}

Retrieved context from the knowledge base:
{context}

User question: {question}

Extract and present the most important ideas. Be ruthless about what matters.
Structure your response clearly. Cite sources inline as [Source: <document name>].
End with a one-sentence "big picture" takeaway.""",

    "challenge_pull": """You are a rigorous problem-set tutor.
The user wants a real textbook problem. {depth_instruction}

Retrieved problems from the knowledge base:
{context}

User request: {question}

Present one problem from the retrieved material exactly as written.
Label it clearly: Problem [Source: <document, page/section>].
Do NOT give the solution yet. 
After presenting the problem, ask: "Ready for a hint, or want to give it a try first?" """,

    "challenge_generate": """You are a creative, pedagogically-sound problem designer.
The user wants a new problem. {depth_instruction}

Reference problem(s) for context:
{context}

User request: {question}
Mode: {challenge_mode}

Generate a new problem that matches the request:
- "similar": same skill and structure, different numbers/scenario
- "simpler": same skill, reduced complexity, more scaffolding
- "harder": same skill, elevated complexity, multi-step or abstract

Label the skill being tested. Do NOT give the solution yet.
After presenting the problem, ask: "Want a hint, or give it a shot?" """,

    "challenge_general": """You are an adaptive quiz master.
The user wants to be tested. {depth_instruction}

Retrieved content from the knowledge base:
{context}

User request: {question}

Generate one focused question that tests understanding of the topic.
Match the depth level strictly. Use the retrieved content as your source.
Do NOT give the answer yet — wait for the user to respond.""",
}

# ─── Router ────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    system_prompt: str
    user_prompt: str
    intent: str
    depth: int
    depth_label: str
    challenge_subtype: Optional[str]

def build_context_string(chunks: list) -> str:
    """Format retrieved chunks into a readable context block."""
    if not chunks:
        return "No relevant content found in the knowledge base."
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "")
        text = chunk.get("text", "")
        page_str = f", p.{page}" if page else ""
        parts.append(f"[{i}] ({source}{page_str})\n{text}")
    
    return "\n\n".join(parts)


def route_prompt(
    classified: dict,
    question: str,
    chunks: list,
    last_problem: Optional[dict] = None
) -> PromptResult:
    """
    Core router — takes classified intent + chunks → returns filled PromptResult.
    This is the programmatic prompt selection that satisfies Story 8.
    """
    intent = classified["intent"]
    depth = classified["depth"]
    challenge_subtype = classified.get("challenge_subtype")
    
    depth_config = DEPTH_CONFIGS[depth]
    depth_instruction = depth_config["instruction"]
    context_str = build_context_string(chunks)

    # ── Challenge routing (most complex branch) ──
    if intent == "challenge":
        if challenge_subtype == "pull":
            template_key = "challenge_pull"
            filled = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=context_str,
                question=question
            )

        elif challenge_subtype in ("similar", "simpler", "harder"):
            template_key = "challenge_generate"
            ref_context = build_context_string([last_problem]) if last_problem else context_str
            filled = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=ref_context,
                question=question,
                challenge_mode=challenge_subtype
            )

        else:
            template_key = "challenge_general"
            filled = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=context_str,
                question=question
            )

    # ── Standard intents ──
    else:
        template_key = intent if intent in PROMPT_TEMPLATES else "explain"
        filled = PROMPT_TEMPLATES[template_key].format(
            depth_instruction=depth_instruction,
            context=context_str,
            question=question
        )

    system = (
        "You are an adaptive learning companion. "
        "You only teach from the provided knowledge base — never invent facts. "
        "Always cite your sources. Adapt your language to the user's level."
    )

    return PromptResult(
        system_prompt=system,
        user_prompt=filled,
        intent=intent,
        depth=depth,
        depth_label=depth_config["label"],
        challenge_subtype=challenge_subtype
    )