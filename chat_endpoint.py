"""
chat_endpoint_v2.py — Story 2 (production-ready)

POST /chat  — natural language questions → grounded, cited responses.

Pipeline:
  classify intent + depth (v2 classifier)
    → apply session-aware depth adjustment
      → retrieve top-k chunks (ChromaDB vector search or BM25 fallback)
        → route to correct prompt template (Story 8)
          → generate grounded response (GPT-4o or local)
            → return structured JSON with sources + grounding metadata

Key improvements over chat_endpoint.py:
  - Uses intent_classifier_v2 (safe import, no bare AsyncOpenAI() at module level)
  - Uses VectorKnowledgeBase (ChromaDB) with BM25 fallback
  - depth_label always populated (was missing in v1 path)
  - Source citations extracted from response text
  - Session depth history properly bounded
  - Challenge "last_problem" stores full chunk, not just first element
  - Graceful error handling: 500s become useful JSON error responses
"""

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from intent_classifier_v2 import apply_session_depth, classify_intent
from prompt_router import route_prompt
from grounding import GroundedResponseEngine
from vector_kb import VectorKnowledgeBase

app = FastAPI(title="Adaptive Learning Companion — Chat API")

USE_LOCAL = os.environ.get("USE_LOCAL_CLASSIFIER", "false").lower() == "true"

# Singletons — created once at startup
kb = VectorKnowledgeBase()
grounded_engine = GroundedResponseEngine()

# In-memory sessions — swap for Redis in production
sessions: Dict[str, Dict[str, Any]] = {}

MAX_HISTORY = 12       # message pairs kept per session
MAX_DEPTH_HISTORY = 20  # depth readings kept for rolling average


# ─── Request / Response models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 3          # caller can request more/fewer chunks


class SourceRef(BaseModel):
    source: str
    page: Optional[int]
    score: float


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    depth: int
    depth_label: str
    challenge_subtype: Optional[str]
    sources: List[SourceRef]
    latency_ms: int
    grounded: bool
    grounding_score: float
    unsupported_spans: List[str]
    retrieval_mode: str              # "vector" or "bm25" — useful for debugging


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str]) -> tuple[str, Dict[str, Any]]:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "history": [],
            "depth_history": [],
            "last_problem": None,
            "created_at": time.time(),
            "query_count": 0,
        }
    return session_id, sessions[session_id]


def _extract_cited_sources(response_text: str) -> List[str]:
    """
    Pull out any [Source: X] citations the LLM included in its answer.
    Used for cross-referencing with retrieved chunks.
    """
    return re.findall(r"\[Source:\s*([^\]]+)\]", response_text, re.IGNORECASE)


def _trim_session(session: Dict[str, Any]) -> None:
    """Keep session history bounded so context window stays manageable."""
    if len(session["history"]) > MAX_HISTORY * 2:
        # Keep the first 2 messages (opening context) + most recent MAX_HISTORY pairs
        session["history"] = session["history"][:2] + session["history"][-(MAX_HISTORY * 2 - 2):]
    if len(session["depth_history"]) > MAX_DEPTH_HISTORY:
        session["depth_history"] = session["depth_history"][-MAX_DEPTH_HISTORY:]


# ─── Global error handler ────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


# ─── Main chat endpoint ──────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.time()
    top_k = max(1, min(request.top_k or 3, 10))  # clamp 1–10

    # 1. Session
    session_id, session = _get_or_create_session(request.session_id)
    session["query_count"] += 1

    # 2. Classify intent + depth
    classified = await classify_intent(
        message=request.message,
        conversation_history=session["history"],
    )

    # Ensure depth_label is always present (safety net)
    classified.setdefault(
        "depth_label",
        ["", "beginner", "intermediate", "advanced"][classified["depth"]]
    )

    # 3. Session-aware depth adjustment
    classified = apply_session_depth(classified, session["depth_history"])

    # 4. Retrieve chunks
    retrieved = kb.search(request.message, top_k=top_k)
    chunks: List[Dict[str, Any]] = [
        {
            "text": r.text,
            "source": r.source,
            "page": r.page,
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in retrieved
    ]

    # 5. Route to prompt template (Story 8)
    prompt_result = route_prompt(
        classified=classified,
        question=request.message,
        chunks=chunks,
        last_problem=session.get("last_problem"),
    )

    # 6. Generate grounded response
    grounded = await grounded_engine.answer(
        question=request.message,
        chunks=chunks,
        routed_prompt=prompt_result.user_prompt,
    )

    # 7. Update session state
    session["history"].append({"role": "user", "content": request.message})
    session["history"].append({"role": "assistant", "content": grounded.answer})
    session["depth_history"].append(classified["depth"])
    _trim_session(session)

    # Store the full chunk as last_problem for challenge follow-up routing
    if (
        classified["intent"] == "challenge"
        and classified.get("challenge_subtype") == "pull"
        and chunks
    ):
        session["last_problem"] = chunks[0]

    latency_ms = int((time.time() - start) * 1000)

    return ChatResponse(
        response=grounded.answer,
        session_id=session_id,
        intent=classified["intent"],
        depth=classified["depth"],
        depth_label=classified["depth_label"],
        challenge_subtype=classified.get("challenge_subtype"),
        sources=[
            SourceRef(
                source=c["source"],
                page=c.get("page"),
                score=c["score"],
            )
            for c in chunks
        ],
        latency_ms=latency_ms,
        grounded=grounded.grounded,
        grounding_score=grounded.grounding_score,
        unsupported_spans=grounded.unsupported_spans,
        retrieval_mode=kb._mode,
    )


# ─── Session info endpoint (useful for developer dashboard) ─────────────────

@app.get("/session/{session_id}")
async def session_info(session_id: str) -> JSONResponse:
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    s = sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "query_count": s["query_count"],
        "history_length": len(s["history"]),
        "depth_history": s["depth_history"],
        "has_last_problem": s["last_problem"] is not None,
        "created_at": s["created_at"],
    })


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> JSONResponse:
    sessions.pop(session_id, None)
    return JSONResponse({"cleared": True})
