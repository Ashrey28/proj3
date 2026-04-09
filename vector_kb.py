"""
vector_kb.py — Story 2 + 5
ChromaDB-backed knowledge base with OpenAI embeddings.
Drop-in replacement for SimpleKnowledgeBase in app.py.

Swap in app.py:
    from vector_kb import VectorKnowledgeBase
    kb = VectorKnowledgeBase()

Requires:
    pip install chromadb openai
    OPENAI_API_KEY env var set (or USE_LOCAL_CLASSIFIER=true for BM25 fallback)
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Any, Dict, List, Optional

# ── Shared result type (same as rag.py so app.py needs no changes) ──────────

@dataclass
class RetrievalResult:
    text: str
    source: str
    page: Optional[int]
    score: float
    metadata: Dict[str, Any]


# ── Try to load ChromaDB + OpenAI; fall back to BM25 if unavailable ─────────

USE_LOCAL = os.environ.get("USE_LOCAL_CLASSIFIER", "false").lower() == "true"

try:
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    USE_LOCAL = True


DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100  # OpenAI max per request


# ════════════════════════════════════════════════════════════════════════════
#  Vector Knowledge Base (ChromaDB + OpenAI embeddings)
# ════════════════════════════════════════════════════════════════════════════

class VectorKnowledgeBase:
    """
    Primary knowledge base for production.
    Uses ChromaDB for vector storage and OpenAI embeddings for similarity search.
    Automatically falls back to BM25 if ChromaDB/OpenAI are unavailable.
    """

    def __init__(self, path: str = DEFAULT_KB_PATH):
        self.path = path
        self.documents: List[Dict[str, Any]] = []

        if _CHROMA_AVAILABLE and not USE_LOCAL:
            self._mode = "vector"
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._embed_client = OpenAI()
        else:
            self._mode = "bm25"
            self._doc_freq: Counter = Counter()
            self._avg_doc_len: float = 0.0

        self.reload()

    # ── Embedding helpers ────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API in batches."""
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i : i + EMBED_BATCH]
            response = self._embed_client.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    # ── BM25 helpers (fallback) ──────────────────────────────────────────────

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        for suffix in ("ing", "ness", "edly", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[: -len(suffix)]
                break
        return token

    def _tokenize(self, text: str) -> List[str]:
        return [self._normalize_token(t) for t in re.findall(r"[a-zA-Z0-9_]+", text.lower())]

    def _rebuild_bm25_index(self) -> None:
        self._doc_freq = Counter()
        total_len = 0
        for doc in self.documents:
            total_len += doc["length"]
            for token in set(doc["tokens"]):
                self._doc_freq[token] += 1
        self._avg_doc_len = (total_len / len(self.documents)) if self.documents else 0.0

    def _bm25_score(self, query_tokens: List[str], doc: Dict[str, Any]) -> float:
        k1, b = 1.5, 0.75
        score = 0.0
        doc_len = doc["length"]
        avg_len = self._avg_doc_len or 1.0
        for token in query_tokens:
            tf = doc["token_counts"].get(token, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = log(1 + ((len(self.documents) - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / avg_len))
            score += idf * ((tf * (k1 + 1)) / denom)
        return score

    # ── Document normalization ───────────────────────────────────────────────

    def _normalize_doc(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        text = str(doc.get("text", "")).strip()
        source = str(doc.get("source", f"document_{index + 1}"))
        page = doc.get("page")
        metadata = dict(doc.get("metadata") or {})
        tokens = self._tokenize(text) if self._mode == "bm25" else []
        return {
            "id": str(doc.get("id", f"doc_{index + 1}")),
            "text": text,
            "source": source,
            "page": page,
            "metadata": metadata,
            "tokens": tokens,
            "token_counts": Counter(tokens),
            "length": len(tokens) or 1,
        }

    # ── Load / reload ────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Re-read knowledge_base.json and re-index everything."""
        if not os.path.exists(self.path):
            self.documents = []
            if self._mode == "bm25":
                self._rebuild_bm25_index()
            return

        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.documents = [
            self._normalize_doc(doc, i)
            for i, doc in enumerate(raw)
            if str(doc.get("text", "")).strip()
        ]

        if self._mode == "bm25":
            self._rebuild_bm25_index()
        else:
            self._sync_to_chroma()

    def _sync_to_chroma(self) -> None:
        """
        Upsert all documents from knowledge_base.json into ChromaDB.
        Uses doc IDs so re-loading is idempotent (no duplicates).
        """
        if not self.documents:
            return

        # Figure out which IDs are already in the collection
        existing = set(self._collection.get(include=[])["ids"])
        to_add = [d for d in self.documents if d["id"] not in existing]

        if not to_add:
            return

        texts = [d["text"] for d in to_add]
        embeddings = self._embed(texts)

        self._collection.add(
            ids=[d["id"] for d in to_add],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": d["source"],
                    "page": d["page"] if d["page"] is not None else -1,
                    **{k: str(v) for k, v in d["metadata"].items()},
                }
                for d in to_add
            ],
        )

    # ── Add documents ────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        """
        Persist new documents to knowledge_base.json and index them.
        Returns count of documents added.
        """
        existing = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(docs)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        self.reload()
        return len(docs)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Return top_k most relevant chunks for the query.
        Uses cosine similarity on OpenAI embeddings (vector mode)
        or BM25 (fallback mode).
        """
        if not self.documents:
            return []

        if self._mode == "vector":
            return self._vector_search(query, top_k)
        else:
            return self._bm25_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query_embedding = self._embed([query])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        output: List[RetrievalResult] = []
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            # Chroma returns cosine *distance* (0=identical, 2=opposite) → convert to similarity
            distance = results["distances"][0][i]
            score = round(1.0 - distance / 2.0, 4)
            page = meta.get("page")
            if page == -1:
                page = None
            output.append(
                RetrievalResult(
                    text=doc_text,
                    source=meta.get("source", "Unknown"),
                    page=page,
                    score=score,
                    metadata={k: v for k, v in meta.items() if k not in ("source", "page")},
                )
            )
        return output

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query_tokens = self._tokenize(query)
        scored: List[RetrievalResult] = []
        for doc in self.documents:
            score = self._bm25_score(query_tokens, doc)
            for token in query_tokens:
                if token and token in doc["text"].lower():
                    score += 0.15
            if score <= 0:
                continue
            scored.append(
                RetrievalResult(
                    text=doc["text"],
                    source=doc["source"],
                    page=doc["page"],
                    score=round(score, 4),
                    metadata=doc["metadata"],
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ── Admin helpers ────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Delete all documents from ChromaDB and the JSON store."""
        if self._mode == "vector":
            self._chroma.delete_collection(COLLECTION_NAME)
            self._collection = self._chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        if os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([], f)
        self.documents = []
        if self._mode == "bm25":
            self._rebuild_bm25_index()
