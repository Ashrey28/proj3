"""
vector_kb.py — Story 2 + 5
ChromaDB-backed knowledge base with OpenAI embeddings.
Falls back to BM25 when local mode is enabled or vector deps are unavailable.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Any, Dict, List, Optional

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
EMBED_BATCH = 100


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: Optional[int]
    score: float
    metadata: Dict[str, Any]


class VectorKnowledgeBase:
    def __init__(self, path: str = DEFAULT_KB_PATH):
        self.path = path
        self.documents: List[Dict[str, Any]] = []
        if _CHROMA_AVAILABLE and not USE_LOCAL:
            self._mode = "vector"
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
            self._collection = self._chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
            self._embed_client = OpenAI()
        else:
            self._mode = "bm25"
            self._doc_freq: Counter = Counter()
            self._avg_doc_len: float = 0.0
        self.reload()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            response = self._embed_client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
        return all_embeddings

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        for suffix in ("ing", "ness", "edly", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[:-len(suffix)]
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

    def _normalize_doc(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        text = str(doc.get("text", "")).strip()
        source = str(doc.get("source", f"document_{index + 1}"))
        page = doc.get("page")
        metadata = dict(doc.get("metadata") or {})
        tokens = self._tokenize(text) if self._mode == "bm25" else []
        default_id = doc.get("id") or metadata.get("chunk_id") or f"doc_{index + 1}"
        metadata.setdefault("chunk_id", str(default_id))
        return {
            "id": str(default_id),
            "text": text,
            "source": source,
            "page": page,
            "metadata": metadata,
            "tokens": tokens,
            "token_counts": Counter(tokens),
            "length": len(tokens) or 1,
        }

    def _load_raw(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, list) else []

    def _save_raw(self, docs: List[Dict[str, Any]]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

    def reload(self) -> None:
        raw = self._load_raw()
        self.documents = [self._normalize_doc(doc, i) for i, doc in enumerate(raw) if str(doc.get("text", "")).strip()]
        if self._mode == "bm25":
            self._rebuild_bm25_index()
        else:
            self._resync_chroma()

    def _resync_chroma(self) -> None:
        self._chroma.delete_collection(COLLECTION_NAME)
        self._collection = self._chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        if not self.documents:
            return
        texts = [d["text"] for d in self.documents]
        embeddings = self._embed(texts)
        self._collection.add(
            ids=[d["id"] for d in self.documents],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": d["source"], "page": d["page"] if d["page"] is not None else -1, **{k: str(v) for k, v in d["metadata"].items()}} for d in self.documents],
        )

    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        existing = self._load_raw()
        start = len(existing)
        normalized = []
        for offset, doc in enumerate(docs, start=1):
            doc = dict(doc)
            metadata = dict(doc.get("metadata") or {})
            doc_id = str(doc.get("id") or metadata.get("chunk_id") or f"doc_{start + offset}")
            doc["id"] = doc_id
            metadata["chunk_id"] = doc_id
            doc["metadata"] = metadata
            normalized.append(doc)
        existing.extend(normalized)
        self._save_raw(existing)
        self.reload()
        return len(normalized)

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        if not self.documents:
            return []
        return self._vector_search(query, top_k) if self._mode == "vector" else self._bm25_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        query_embedding = self._embed([query])[0]
        results = self._collection.query(query_embeddings=[query_embedding], n_results=min(top_k, self._collection.count() or 1), include=["documents", "metadatas", "distances"])
        output: List[RetrievalResult] = []
        for i, doc_text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = round(1.0 - distance / 2.0, 4)
            page = meta.get("page")
            if page == -1:
                page = None
            output.append(RetrievalResult(text=doc_text, source=meta.get("source", "Unknown"), page=page, score=score, metadata={k: v for k, v in meta.items() if k not in ("source", "page")}))
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
            scored.append(RetrievalResult(text=doc["text"], source=doc["source"], page=doc["page"], score=round(score, 4), metadata=doc["metadata"]))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        if self._mode == "vector":
            self._chroma.delete_collection(COLLECTION_NAME)
            self._collection = self._chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        self._save_raw([])
        self.documents = []
        if self._mode == "bm25":
            self._rebuild_bm25_index()

    def replace_all_documents(self, docs: List[Dict[str, Any]]) -> int:
        self.clear()
        return self.add_documents(docs) if docs else 0

    def delete_document(self, doc_id: str) -> bool:
        raw = self._load_raw()
        kept = []
        deleted = False
        for index, doc in enumerate(raw):
            current_id = str(doc.get("id") or (doc.get("metadata") or {}).get("chunk_id") or f"doc_{index + 1}")
            if current_id == str(doc_id):
                deleted = True
                continue
            kept.append(doc)
        if not deleted:
            return False
        self._save_raw(kept)
        self.reload()
        return True

    def update_document(self, doc_id: str, updated_doc: Dict[str, Any]) -> bool:
        raw = self._load_raw()
        changed = False
        for index, doc in enumerate(raw):
            current_id = str(doc.get("id") or (doc.get("metadata") or {}).get("chunk_id") or f"doc_{index + 1}")
            if current_id == str(doc_id):
                metadata = dict(doc.get("metadata") or {})
                metadata.update(updated_doc.get("metadata") or {})
                metadata["chunk_id"] = str(doc_id)
                raw[index] = {
                    "id": str(doc_id),
                    "source": updated_doc.get("source", doc.get("source", f"document_{index + 1}")),
                    "text": updated_doc.get("text", doc.get("text", "")),
                    "page": updated_doc.get("page", doc.get("page")),
                    "metadata": metadata,
                }
                changed = True
                break
        if not changed:
            return False
        self._save_raw(raw)
        self.reload()
        return True
