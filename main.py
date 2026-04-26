import json
import os
import re
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data_preprocessor import DataPreprocessor
from evaluation import completion_score, source_hit, summarize_results, token_f1
from grounding import GroundedResponseEngine
from intent_classifier_v2 import apply_session_depth, classify_intent
from prompt_router import route_prompt
from rag import EvaluationStore
from vector_kb import VectorKnowledgeBase

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="EduRAG")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

kb = VectorKnowledgeBase()
evaluation_store = EvaluationStore()
grounded_engine = GroundedResponseEngine()
sessions: Dict[str, Dict[str, Any]] = {}

MIN_RELEVANCE_SCORE = 0.45
MIN_TOP_SCORE_GAP = 0.03

MAX_ADMIN_LOGS = 250
admin_logs: Deque[Dict[str, Any]] = deque(maxlen=MAX_ADMIN_LOGS)
admin_stats: Dict[str, Any] = {
    "started_at": datetime.now(timezone.utc).isoformat(),
    "chat_requests": 0,
    "chat_successes": 0,
    "chat_failures": 0,
    "admin_actions": 0,
    "dataset_mutations": 0,
    "total_chat_latency_ms": 0,
    "last_reload_at": None,
    "last_clear_at": None,
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how",
    "i", "in", "is", "it", "like", "me", "my", "of", "on", "or", "our", "should",
    "tell", "that", "the", "their", "them", "there", "these", "this", "to", "today",
    "was", "we", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_admin_event(
    event_type: str,
    status: str = "success",
    latency_ms: Optional[int] = None,
    **details: Any,
) -> None:
    entry: Dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "event_type": event_type,
        "status": status,
    }
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    for key, value in details.items():
        if value is not None:
            entry[key] = value
    admin_logs.appendleft(entry)


def admin_metrics_payload() -> Dict[str, Any]:
    total = admin_stats["chat_requests"]
    successes = admin_stats["chat_successes"]
    avg_latency = round(admin_stats["total_chat_latency_ms"] / successes, 1) if successes else 0.0
    success_rate = round((successes / total) * 100, 1) if total else 0.0

    return {
        "started_at": admin_stats["started_at"],
        "documents": len(kb.documents),
        "chat_requests": total,
        "chat_successes": successes,
        "chat_failures": admin_stats["chat_failures"],
        "success_rate_pct": success_rate,
        "avg_chat_latency_ms": avg_latency,
        "admin_actions": admin_stats["admin_actions"],
        "dataset_mutations": admin_stats["dataset_mutations"],
        "last_reload_at": admin_stats["last_reload_at"],
        "last_clear_at": admin_stats["last_clear_at"],
        "logs": list(admin_logs),
    }


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    intent_override: Optional[str] = None
    depth_override: Optional[int] = None
    step_by_step: bool = False


class IngestRequest(BaseModel):
    source: str
    text: str
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateDocumentRequest(BaseModel):
    source: str
    text: str
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ReplaceDatasetRequest(BaseModel):
    documents: List[Dict[str, Any]]


class ExamRequest(BaseModel):
    chapters: str
    mcq: int = 3
    frq: int = 2
    true_false: int = 0
    math: int = 0


def filter_retrieved_results(message: str, retrieved: List[Any]) -> List[Any]:
    if not retrieved:
        return []

    filtered = []
    top_score = retrieved[0].score
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9_]+", message.lower())
        if len(term) > 2 and term not in STOPWORDS
    }

    for item in retrieved:
        text_terms = set(re.findall(r"[a-z0-9_]+", item.text.lower()))
        overlap_count = len(query_terms & text_terms)
        has_lexical_support = overlap_count > 0
        if item.score >= MIN_RELEVANCE_SCORE and (top_score - item.score) <= 0.25 and has_lexical_support:
            filtered.append(item)

    return filtered


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/developer")
async def developer_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "developer.html"))


@app.get("/admin")
async def admin_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


@app.get("/exam")
async def exam_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "exam.html"))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "documents": len(kb.documents)}


@app.get("/api/dataset")
async def dataset_status() -> Dict[str, Any]:
    preview = []
    for document in kb.documents[:50]:
        preview.append(
            {
                "id": document["id"],
                "source": document["source"],
                "page": document["page"],
                "text_preview": document["text"][:180],
            }
        )
    return {"count": len(kb.documents), "preview": preview}


@app.post("/api/dataset/reload")
async def reload_dataset() -> Dict[str, Any]:
    start = time.time()
    kb.reload()
    admin_stats["admin_actions"] += 1
    admin_stats["last_reload_at"] = utc_now_iso()
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event("dataset_reload", latency_ms=latency_ms, count=len(kb.documents))
    return {"reloaded": True, "count": len(kb.documents)}


@app.post("/api/dataset/clear")
async def clear_dataset() -> Dict[str, Any]:
    start = time.time()
    kb.clear()
    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    admin_stats["last_clear_at"] = utc_now_iso()
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event("dataset_clear", latency_ms=latency_ms, count=len(kb.documents))
    return {"cleared": True, "count": len(kb.documents)}


@app.post("/api/dataset/add")
async def add_document(request: IngestRequest) -> Dict[str, Any]:
    start = time.time()
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    added = kb.add_documents(
        [
            {
                "source": request.source,
                "text": request.text,
                "page": request.page,
                "metadata": request.metadata or {},
            }
        ]
    )
    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event(
        "dataset_add",
        latency_ms=latency_ms,
        count=added,
        source=request.source,
        page=request.page,
    )
    return {"added": added, "count": len(kb.documents)}


@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    start = time.time()
    contents = await file.read()

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not parse PDF.") from exc

    raw_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            raw_pages.append(
                {
                    "source": file.filename,
                    "text": text,
                    "page": i + 1,
                    "metadata": {},
                }
            )

    if not raw_pages:
        raise HTTPException(status_code=422, detail="No extractable text found. The PDF may be scanned.")

    preprocessor = DataPreprocessor(chunk_size=800, chunk_overlap=100)
    chunks = preprocessor.preprocess_batch(raw_pages)
    added = kb.add_documents(chunks)

    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event(
        "pdf_upload",
        latency_ms=latency_ms,
        count=added,
        source=file.filename,
        pages=len(raw_pages),
    )

    return {"added": added, "count": len(kb.documents), "pages": len(raw_pages)}


class ScrapeRequest(BaseModel):
    url: str


@app.post("/api/ingest/url")
async def ingest_url(request: ScrapeRequest) -> Dict[str, Any]:
    """User Story 6: Web scraping ingestion method."""
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(request.url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (compatible; EduRAG/1.0)"
        })
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {str(e)}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text content found at that URL.")

    preprocessor = DataPreprocessor(chunk_size=800, chunk_overlap=100)
    chunks = preprocessor.preprocess(text, source=request.url)
    chunks = [c for c in chunks if preprocessor.validate_chunk(c["text"])]

    added = kb.add_documents(chunks)
    return {"added": added, "count": len(kb.documents), "source": request.url}

@app.put("/api/dataset/{doc_id}")
async def update_document(doc_id: str, request: UpdateDocumentRequest) -> Dict[str, Any]:
    start = time.time()
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    updated = kb.update_document(
        doc_id,
        {
            "source": request.source,
            "text": request.text,
            "page": request.page,
            "metadata": request.metadata or {},
        },
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Document not found.")

    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event(
        "dataset_update",
        latency_ms=latency_ms,
        id=doc_id,
        source=request.source,
        page=request.page,
    )
    return {"updated": True, "count": len(kb.documents)}


@app.delete("/api/dataset/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, Any]:
    start = time.time()
    deleted = kb.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")

    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event("dataset_delete", latency_ms=latency_ms, id=doc_id)
    return {"deleted": True, "count": len(kb.documents)}


@app.post("/api/dataset/replace")
async def replace_dataset(request: ReplaceDatasetRequest) -> Dict[str, Any]:
    start = time.time()
    count = kb.replace_all_documents(request.documents)
    admin_stats["admin_actions"] += 1
    admin_stats["dataset_mutations"] += 1
    latency_ms = int((time.time() - start) * 1000)
    record_admin_event("dataset_replace", latency_ms=latency_ms, count=count)
    return {"replaced": True, "count": count}


@app.get("/api/admin/metrics")
async def admin_metrics() -> Dict[str, Any]:
    return admin_metrics_payload()


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    start = time.time()
    admin_stats["chat_requests"] += 1
    session_id = request.session_id or str(uuid.uuid4())

    try:
        session = sessions.setdefault(session_id, {"history": [], "depth_history": [], "last_problem": None})

        classified = await classify_intent(request.message, session["history"])
        classified = apply_session_depth(classified, session["depth_history"])
        if "depth_label" not in classified:
            classified["depth_label"] = ["", "beginner", "intermediate", "advanced"][classified["depth"]]

        if request.depth_override and request.depth_override in (1, 2, 3):
            classified["depth"] = request.depth_override
            classified["depth_label"] = ["", "beginner", "intermediate", "advanced"][request.depth_override]

        mode_to_intent = {
            "quiz": "challenge",
            "explore": "explore",
            "summarize": "summarize",
            "studyguide": "study_guide",
            "learn": None,
        }
        if request.intent_override and request.intent_override in mode_to_intent:
            forced = mode_to_intent[request.intent_override]
            if forced:
                classified["intent"] = forced
                if forced != "challenge":
                    classified["challenge_subtype"] = None

        retrieved = filter_retrieved_results(request.message, kb.search(request.message, top_k=3))
        chunks = [
            {
                "text": item.text,
                "source": item.source,
                "page": item.page,
                "score": round(item.score, 4),
                "metadata": item.metadata,
            }
            for item in retrieved
        ]

        prompt_result = route_prompt(
            classified=classified,
            question=request.message,
            chunks=chunks,
            last_problem=session.get("last_problem"),
        )

        user_prompt = prompt_result.user_prompt
        if request.step_by_step:
            user_prompt += "\n\nIMPORTANT: Do NOT just give the final answer. Walk through the solution step by step. Number each step. For each step, explain what you are doing and why before moving on to the next."

        grounded = await grounded_engine.answer(
            request.message,
            chunks,
            user_prompt,
            router_persona=prompt_result.system_prompt,
        )

        session["history"].append({"role": "user", "content": request.message})
        session["history"].append({"role": "assistant", "content": grounded.answer})
        session["depth_history"].append(classified["depth"])

        if classified["intent"] == "challenge" and classified.get("challenge_subtype") == "pull" and chunks:
            session["last_problem"] = chunks[0]

        latency_ms = int((time.time() - start) * 1000)
        admin_stats["chat_successes"] += 1
        admin_stats["total_chat_latency_ms"] += latency_ms
        record_admin_event(
            "chat_query",
            latency_ms=latency_ms,
            query=request.message,
            session_id=session_id,
            source_count=len(chunks),
            grounded=grounded.grounded,
        )

        return JSONResponse(
            {
                "response": grounded.answer,
                "session_id": session_id,
                "intent": classified["intent"],
                "depth": classified["depth"],
                "depth_label": classified["depth_label"],
                "challenge_subtype": classified.get("challenge_subtype"),
                "sources": [{"source": c["source"], "page": c.get("page"), "score": c["score"]} for c in chunks],
                "latency_ms": latency_ms,
                "grounded": grounded.grounded,
                "grounding_score": grounded.grounding_score,
                "unsupported_spans": grounded.unsupported_spans,
            }
        )

    except Exception as exc:
        latency_ms = int((time.time() - start) * 1000)
        admin_stats["chat_failures"] += 1
        record_admin_event(
            "chat_query",
            status="failure",
            latency_ms=latency_ms,
            query=request.message,
            session_id=session_id,
            error=str(exc),
        )
        raise


@app.post("/api/evaluate")
async def evaluate() -> Dict[str, Any]:
    cases = evaluation_store.load()
    if not cases:
        raise HTTPException(status_code=400, detail="No evaluation cases found in data/evaluation_set.json")

    results: List[Dict[str, Any]] = []
    for case in cases:
        question = str(case.get("question", "")).strip()
        expected_answer = str(case.get("expected_answer", "")).strip()
        expected_sources = list(case.get("expected_sources", []))
        if not question or not expected_answer:
            continue

        classified = await classify_intent(question, [])
        if "depth_label" not in classified:
            classified["depth_label"] = ["", "beginner", "intermediate", "advanced"][classified["depth"]]

        retrieved = filter_retrieved_results(question, kb.search(question, top_k=3))
        chunks = [
            {
                "text": item.text,
                "source": item.source,
                "page": item.page,
                "score": round(item.score, 4),
                "metadata": item.metadata,
            }
            for item in retrieved
        ]

        prompt_result = route_prompt(
            classified=classified,
            question=question,
            chunks=chunks,
            last_problem=None,
        )
        grounded = await grounded_engine.answer(
            question,
            chunks,
            prompt_result.user_prompt,
            router_persona=prompt_result.system_prompt,
        )

        f1 = token_f1(grounded.answer, expected_answer)
        answer_correct = (
            f1 >= 0.25
            or expected_answer.lower() in grounded.answer.lower()
            or (grounded.grounding_score > 0.85 and grounded.grounded)
        )
        source_correct = source_hit(chunks, expected_sources) if expected_sources else len(chunks) == 0

        row = {
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": grounded.answer,
            "token_f1": round(f1, 4),
            "answer_correct": answer_correct,
            "source_correct": source_correct,
            "grounded": grounded.grounded,
            "grounding_score": grounded.grounding_score,
            "sources": [{"source": c["source"], "page": c.get("page")} for c in chunks],
        }
        row["completion"] = completion_score(row)
        results.append(row)

    return {"summary": summarize_results(results), "results": results}


@app.post("/api/exam")
async def generate_exam(request: ExamRequest) -> JSONResponse:
    from grounding import USE_LOCAL

    total = request.mcq + request.frq + request.true_false + request.math
    if total == 0:
        raise HTTPException(status_code=400, detail="Specify at least one question.")

    retrieved = kb.search(f"chapter {request.chapters} key concepts", top_k=10)
    if not retrieved:
        raise HTTPException(status_code=422, detail="No content found for those chapters.")

    context = "\n\n".join(
        f"[{i + 1}] ({r.source}{f', p.{r.page}' if r.page else ''})\n{r.text}"
        for i, r in enumerate(retrieved)
    )

    math_context = ""
    if request.math:
        math_results = kb.search(
            f"chapter {request.chapters} equation formula derivation calculate solve problem example",
            top_k=8,
        )
        if math_results:
            math_context = (
                "\n\nMATH SOURCE MATERIAL (worked examples and equations from the textbook — adapt these for math problems):\n"
                + "\n\n".join(
                    f"[M{i + 1}] ({r.source}{f', p.{r.page}' if r.page else ''})\n{r.text}"
                    for i, r in enumerate(math_results)
                )
            )

    breakdown = []
    if request.mcq:
        breakdown.append(f"{request.mcq} multiple-choice (4 options A-D, include 'answer' key with correct letter)")
    if request.frq:
        breakdown.append(f"{request.frq} free-response (include 'answer' key with model answer)")
    if request.true_false:
        breakdown.append(f"{request.true_false} true/false (include 'answer' key: true or false)")
    if request.math:
        breakdown.append(
            f"{request.math} math problems: take specific equations or worked examples from the MATH SOURCE MATERIAL and create a new problem by modifying the scenario, changing variable values, or asking for a different step. "
            f"Show all relevant equations using LaTeX (use \\\\( ... \\\\) for inline math and \\\\[ ... \\\\] for display math). "
            f"Include a full step-by-step solution in the 'answer' field, also in LaTeX."
        )

    prompt = f"""Generate a practice exam on: {request.chapters}
Breakdown: {', '.join(breakdown)}. Total: {total} questions. Use ONLY the source material below.

{context}{math_context}

Return JSON exactly:
{{"title": "Practice Exam — <topic>", "questions": [
  {{"type": "mcq", "question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "explanation": "..."}},
  {{"type": "frq", "question": "...", "answer": "..."}},
  {{"type": "tf", "question": "...", "answer": true}},
  {{"type": "math", "question": "State the problem with equations in LaTeX...", "answer": "Step-by-step solution in LaTeX..."}}
]}}"""

    if USE_LOCAL or grounded_engine.client is None:
        raise HTTPException(status_code=503, detail="Exam generation requires an OpenAI API key.")

    try:
        resp = await grounded_engine.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a rigorous exam generator. Only use provided material. Return valid JSON only. For math problems, write all equations in proper LaTeX using \\( ... \\) for inline and \\[ ... \\] for display math.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3500,
            response_format={"type": "json_object"},
        )
        return JSONResponse(json.loads(resp.choices[0].message.content or "{}"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
