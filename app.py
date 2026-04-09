import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from evaluation import source_hit, summarize_results, token_f1
from grounding import GroundedResponseEngine
from intent_classifier_v2 import apply_session_depth, classify_intent
from prompt_router import route_prompt
from vector_kb import VectorKnowledgeBase
from rag import EvaluationStore
import fitz  # pip install pymupdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from data_preprocessor import DataPreprocessor

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    all_chunks = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            all_chunks.append({
                "source": file.filename,
                "text": text,
                "page": i + 1,
                "metadata": {}
            })
    added = kb.add_documents(all_chunks)
    return {"added": added, "count": len(kb.documents)}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")

app = FastAPI(title="EduRAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

kb = VectorKnowledgeBase()
evaluation_store = EvaluationStore()
grounded_engine = GroundedResponseEngine()
sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    source: str
    text: str
    page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/developer")
async def developer_page() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "developer.html"))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "documents": len(kb.documents)}


@app.get("/api/dataset")
async def dataset_status() -> Dict[str, Any]:
    preview = []
    for document in kb.documents[:10]:
        preview.append(
            {
                "source": document["source"],
                "page": document["page"],
                "text_preview": document["text"][:180],
            }
        )
    return {"count": len(kb.documents), "preview": preview}


@app.post("/api/dataset/reload")
async def reload_dataset() -> Dict[str, Any]:
    kb.reload()
    return {"reloaded": True, "count": len(kb.documents)}


@app.post("/api/dataset/add")
async def add_document(request: IngestRequest) -> Dict[str, Any]:
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
    return {"added": added, "count": len(kb.documents)}

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    import fitz  # pip install pymupdf
    contents = await file.read()
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse PDF.")
    
    raw_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            raw_pages.append({
                "source": file.filename,
                "text": text,
                "page": i + 1,
                "metadata": {}
            })
    
    if not raw_pages:
        raise HTTPException(status_code=422, detail="No extractable text found. The PDF may be scanned.")
    
    preprocessor = DataPreprocessor()
    chunks = preprocessor.preprocess_batch(raw_pages)
    added = kb.add_documents(chunks)
    return {"added": added, "count": len(kb.documents), "pages": len(raw_pages)}


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    start = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.setdefault(session_id, {"history": [], "depth_history": [], "last_problem": None})

    classified = await classify_intent(request.message, session["history"])
    classified = apply_session_depth(classified, session["depth_history"])
    if "depth_label" not in classified:
        classified["depth_label"] = ["", "beginner", "intermediate", "advanced"][classified["depth"]]

    retrieved = kb.search(request.message, top_k=3)
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

    grounded = await grounded_engine.answer(request.message, chunks, prompt_result.user_prompt)

    session["history"].append({"role": "user", "content": request.message})
    session["history"].append({"role": "assistant", "content": grounded.answer})
    session["depth_history"].append(classified["depth"])

    if classified["intent"] == "challenge" and classified.get("challenge_subtype") == "pull" and chunks:
        session["last_problem"] = chunks[0]

    latency_ms = int((time.time() - start) * 1000)
    payload = {
        "response": grounded.answer,
        "session_id": session_id,
        "intent": classified["intent"],
        "depth": classified["depth"],
        "depth_label": classified["depth_label"],
        "challenge_subtype": classified.get("challenge_subtype"),
        "sources": [{"source": chunk["source"], "page": chunk.get("page"), "score": chunk["score"]} for chunk in chunks],
        "latency_ms": latency_ms,
        "grounded": grounded.grounded,
        "grounding_score": grounded.grounding_score,
        "unsupported_spans": grounded.unsupported_spans,
    }
    return JSONResponse(payload)


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

        retrieved = kb.search(question, top_k=3)
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

        prompt_result = route_prompt(classified=classified, question=question, chunks=chunks, last_problem=None)
        grounded = await grounded_engine.answer(question, chunks, prompt_result.user_prompt)
        f1 = token_f1(grounded.answer, expected_answer)
        answer_correct = f1 >= 0.25 or expected_answer.lower() in grounded.answer.lower()
        source_correct = source_hit(chunks, expected_sources) if expected_sources else True

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": grounded.answer,
                "token_f1": round(f1, 4),
                "answer_correct": answer_correct,
                "source_correct": source_correct,
                "grounded": grounded.grounded,
                "grounding_score": grounded.grounding_score,
                "sources": [{"source": chunk["source"], "page": chunk.get("page")} for chunk in chunks],
            }
        )

    return {"summary": summarize_results(results), "results": results}
