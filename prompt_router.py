"""
prompt_router_v2.py — Story 8 (hardened)

Takes classified intent + depth + retrieved chunks → returns filled PromptResult.
This is the programmatic prompt selection that satisfies Story 8.

Changes from prompt_router.py:
  - Safe format() calls: all templates pre-checked, no KeyError on missing fields
  - build_prompt_debug_info() helper for developer dashboard (Story 9)
  - DEPTH_CONFIGS exposed as a public constant for UI rendering
  - Handles empty / None chunks gracefully
  - challenge_generate: falls back to context_str if last_problem is empty dict
  - Unknown intents log a warning and fall back to "explain"
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Depth configuration ─────────────────────────────────────────────────────

DEPTH_CONFIGS: Dict[int, Dict[str, str]] = {
    1: {
        "label": "beginner",
        "instruction": (
            "Use simple, everyday language. Avoid jargon entirely. "
            "Use concrete analogies and real-world examples. "
            "Short sentences. Assume no prior knowledge."
        ),
    },
    2: {
        "label": "intermediate",
        "instruction": (
            "Use standard domain vocabulary but briefly clarify terms when first introduced. "
            "Balance intuition with precision. "
            "Examples are welcome but not required for every point."
        ),
    },
    3: {
        "label": "advanced",
        "instruction": (
            "Use precise technical language freely. Assume solid domain background. "
            "Prioritize depth, nuance, and connections to related concepts. "
            "Minimal hand-holding."
        ),
    },
}


# ─── Prompt templates ────────────────────────────────────────────────────────

PROMPT_TEMPLATES: Dict[str, str] = {

    
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

    "study_guide": """You are an expert educational content designer.
The user wants a structured study guide. {depth_instruction}

Retrieved context from the knowledge base:
{context}

Topic: {question}

Create a comprehensive study guide for this topic with the following sections:
1. **Key Concepts** — list 3-5 core ideas with brief definitions
2. **Detailed Explanations** — expand on each concept with examples
3. **Common Questions & Misconceptions** — highlight what students typically struggle with
4. **Practice Patterns** — describe the types of problems or questions to expect
5. **Study Tips** — actionable advice for mastering this topic
6. **Quick Reference** — a cheat-sheet summary of the main points

Ground everything in the retrieved context. Cite sources as [Source: <document name>].
Make it practical and learning-focused, not just informational.""",
}

# Intents that map directly to a template key
_DIRECT_INTENTS = {"define", "explain", "explore", "clarify", "summarize", "study_guide"}


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    system_prompt: str
    user_prompt: str
    intent: str
    depth: int
    depth_label: str
    challenge_subtype: Optional[str]
    template_key: str                    # which template fired — useful for logging
    context_chunk_count: int             # how many chunks were injected
    debug: Dict[str, Any] = field(default_factory=dict)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_context_string(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered, source-labeled context block."""
    if not chunks:
        return "No relevant content found in the knowledge base."

    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source") or "Unknown"
        page = chunk.get("page")
        text = (chunk.get("text") or "").strip()
        page_str = f", p.{page}" if page is not None else ""
        parts.append(f"[{i}] ({source}{page_str})\n{text}")

    return "\n\n".join(parts)


def build_prompt_debug_info(result: "PromptResult") -> Dict[str, Any]:
    """
    Return a structured summary of what the router decided.
    Intended for the developer dashboard (Story 9) — attach to response payload.
    """
    return {
        "template_key": result.template_key,
        "intent": result.intent,
        "depth": result.depth,
        "depth_label": result.depth_label,
        "challenge_subtype": result.challenge_subtype,
        "context_chunk_count": result.context_chunk_count,
        "system_prompt_length": len(result.system_prompt),
        "user_prompt_length": len(result.user_prompt),
        "user_prompt_preview": result.user_prompt[:300] + ("..." if len(result.user_prompt) > 300 else ""),
    }


# ─── Router ──────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an adaptive learning companion. "
    "You only teach from the provided knowledge base — never invent facts. "
    "Always cite your sources inline as [Source: <document name>]. "
    "Adapt your language to the user's level."
)


def route_prompt(
    classified: Dict[str, Any],
    question: str,
    chunks: List[Dict[str, Any]],
    last_problem: Optional[Dict[str, Any]] = None,
) -> PromptResult:
    """
    Core router — takes classified intent + retrieved chunks → PromptResult.

    Args:
        classified:    Output of classify_intent() — must contain 'intent' and 'depth'.
        question:      The raw user message.
        chunks:        Retrieved knowledge base chunks (list of dicts with 'text', 'source', 'page').
        last_problem:  The chunk used for the previous challenge/pull, if any.

    Returns:
        PromptResult with filled system_prompt, user_prompt, and metadata.
    """
    intent: str = classified.get("intent", "explain")
    depth: int = max(1, min(3, int(classified.get("depth", 2))))
    challenge_subtype: Optional[str] = classified.get("challenge_subtype")

    depth_config = DEPTH_CONFIGS[depth]
    depth_instruction = depth_config["instruction"]
    context_str = build_context_string(chunks)
    chunk_count = len(chunks)

    # ── Challenge routing ────────────────────────────────────────────────────
    if intent == "challenge":

        if challenge_subtype == "pull":
            template_key = "challenge_pull"
            user_prompt = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=context_str,
                question=question,
            )

        elif challenge_subtype in ("similar", "simpler", "harder"):
            template_key = "challenge_generate"
            # Use last_problem if it has text; otherwise fall back to retrieved chunks
            ref_chunks = (
                [last_problem]
                if last_problem and last_problem.get("text")
                else chunks
            )
            ref_context = build_context_string(ref_chunks)
            user_prompt = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=ref_context,
                question=question,
                challenge_mode=challenge_subtype,
            )

        else:
            # "general" or any unrecognised subtype
            template_key = "challenge_general"
            user_prompt = PROMPT_TEMPLATES[template_key].format(
                depth_instruction=depth_instruction,
                context=context_str,
                question=question,
            )

    # ── Standard intents ─────────────────────────────────────────────────────
    else:
        if intent not in _DIRECT_INTENTS:
            warnings.warn(
                f"Unknown intent '{intent}' received by prompt router — "
                "falling back to 'explain'.",
                stacklevel=2,
            )
            intent = "explain"

        template_key = intent
        user_prompt = PROMPT_TEMPLATES[template_key].format(
            depth_instruction=depth_instruction,
            context=context_str,
            question=question,
        )

    result = PromptResult(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        intent=intent,
        depth=depth,
        depth_label=depth_config["label"],
        challenge_subtype=challenge_subtype,
        template_key=template_key,
        context_chunk_count=chunk_count,
    )
    result.debug = build_prompt_debug_info(result)
    return result
