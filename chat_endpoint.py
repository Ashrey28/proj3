"""
Chat endpoint — Story 2
POST /chat — natural language questions, grounded responses, source citations.
Ties together: intent classifier → prompt router → RAG retriever → LLM response.

Set USE_LOCAL_CLASSIFIER=true to run fully offline with no API key.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import time
import uuid
from openai import AsyncOpenAI

from intent_classifier import classify_intent, apply_session_depth
from prompt_router import route_prompt

app = FastAPI()

USE_LOCAL = os.environ.get("USE_LOCAL_CLASSIFIER", "false").lower() == "true"
client = None
if not USE_LOCAL:
    client = AsyncOpenAI()

# In-memory session store (swap for Redis in production)
sessions: dict = {}


# ─── Request / Response models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    depth: int
    depth_label: str
    challenge_subtype: Optional[str]
    sources: list
    latency_ms: int


# ─── Mock retriever (swap for real ChromaDB retriever) ──────────────────────

async def retrieve_chunks(query: str, intent: str, classified: dict) -> list:
    """
    Placeholder retriever — replace with real ChromaDB call.
    In production:
      - For challenge/pull: filter by metadata (topic, skills, difficulty)
      - For all others: semantic similarity search
    """
    return [
        {
            "text": f"[Mock chunk for '{query}' — replace with ChromaDB retrieval]",
            "source": "textbook.pdf",
            "page": 42,
            "topic": classified.get("topic", ""),
            "chunk_type": "problem" if intent == "challenge" else "content"
        }
    ]


# ─── Session helpers ────────────────────────────────────────────────────────

def get_or_create_session(session_id: Optional[str]) -> tuple[str, dict]:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "history": [],
            "depth_history": [],
            "last_problem": None
        }
    return session_id, sessions[session_id]


# ─── Main chat endpoint ──────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()

    session_id, session = get_or_create_session(request.session_id)

    # 1. Classify intent + depth
    classified = await classify_intent(
        message=request.message,
        conversation_history=session["history"]
    )

    # 2. Apply session-aware depth adjustment
    classified = apply_session_depth(classified, session["depth_history"])

    # 3. Retrieve relevant chunks
    chunks = await retrieve_chunks(
        query=request.message,
        intent=classified["intent"],
        classified=classified
    )

    # 4. Route to correct prompt template
    prompt_result = route_prompt(
        classified=classified,
        question=request.message,
        chunks=chunks,
        last_problem=session.get("last_problem")
    )

    # 5. Generate response
    if USE_LOCAL:
        response_text = (
            f"[LOCAL MODE — no API call made]\n\n"
            f"Intent detected: {prompt_result.intent}\n"
            f"Depth: {prompt_result.depth} ({prompt_result.depth_label})\n"
            f"Challenge subtype: {prompt_result.challenge_subtype or 'n/a'}\n\n"
            f"Prompt template that would fire:\n{prompt_result.user_prompt[:400]}..."
        )
    else:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_result.system_prompt},
                *session["history"][-6:],
                {"role": "user", "content": prompt_result.user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        response_text = completion.choices[0].message.content

    # 6. Update session state
    session["history"].append({"role": "user", "content": request.message})
    session["history"].append({"role": "assistant", "content": response_text})
    session["depth_history"].append(classified["depth"])

    if classified["intent"] == "challenge" and classified.get("challenge_subtype") == "pull":
        session["last_problem"] = chunks[0] if chunks else None

    latency = int((time.time() - start) * 1000)

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        intent=classified["intent"],
        depth=classified["depth"],
        depth_label=classified["depth_label"],
        challenge_subtype=classified.get("challenge_subtype"),
        sources=[
            {"source": c.get("source"), "page": c.get("page")}
            for c in chunks
        ],
        latency_ms=latency
    )