import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from math import log
from typing import Any, Dict, List, Optional


DEFAULT_KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.json")
DEFAULT_EVAL_PATH = os.path.join(os.path.dirname(__file__), "data", "evaluation_set.json")


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: Optional[int]
    score: float
    metadata: Dict[str, Any]


class SimpleKnowledgeBase:
    def __init__(self, path: str = DEFAULT_KB_PATH):
        self.path = path
        self.documents: List[Dict[str, Any]] = []
        self._doc_freq: Counter = Counter()
        self._avg_doc_len: float = 0.0
        self.reload()

    def _normalize_token(self, token: str) -> str:
        token = token.lower()
        for suffix in ("ing", "ness", "edly", "edly", "edly", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                token = token[: -len(suffix)]
                break
        return token

    def _tokenize(self, text: str) -> List[str]:
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [self._normalize_token(token) for token in raw_tokens]

    def _normalize_document(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        text = str(doc.get("text", "")).strip()
        source = str(doc.get("source", f"document_{index + 1}"))
        page = doc.get("page")
        metadata = dict(doc.get("metadata", {}))
        tokens = self._tokenize(text)
        return {
            "id": doc.get("id", f"doc_{index + 1}"),
            "text": text,
            "source": source,
            "page": page,
            "metadata": metadata,
            "tokens": tokens,
            "token_counts": Counter(tokens),
            "length": len(tokens) or 1,
        }

    def _rebuild_index(self) -> None:
        self._doc_freq = Counter()
        total_len = 0
        for document in self.documents:
            total_len += document["length"]
            for token in set(document["tokens"]):
                self._doc_freq[token] += 1
        self._avg_doc_len = (total_len / len(self.documents)) if self.documents else 0.0

    def reload(self) -> None:
        if not os.path.exists(self.path):
            self.documents = []
            self._rebuild_index()
            return

        with open(self.path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        self.documents = [self._normalize_document(doc, i) for i, doc in enumerate(raw) if str(doc.get("text", "")).strip()]
        self._rebuild_index()

    def add_documents(self, docs: List[Dict[str, Any]]) -> int:
        existing = []
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        existing.extend(docs)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)
        self.reload()
        return len(docs)

    def bm25_score(self, query_tokens: List[str], document: Dict[str, Any]) -> float:
        if not query_tokens or not self.documents:
            return 0.0

        score = 0.0
        k1 = 1.5
        b = 0.75
        doc_len = document["length"]
        avg_len = self._avg_doc_len or 1.0

        for token in query_tokens:
            tf = document["token_counts"].get(token, 0)
            if tf == 0:
                continue
            df = self._doc_freq.get(token, 0)
            idf = log(1 + ((len(self.documents) - df + 0.5) / (df + 0.5)))
            denom = tf + k1 * (1 - b + b * (doc_len / avg_len))
            score += idf * ((tf * (k1 + 1)) / denom)

        return score

    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        query_tokens = self._tokenize(query)
        scored: List[RetrievalResult] = []

        for document in self.documents:
            score = self.bm25_score(query_tokens, document)
            raw_text = document["text"].lower()
            for token in query_tokens:
                if token and token in raw_text:
                    score += 0.15
            if score <= 0:
                continue
            scored.append(
                RetrievalResult(
                    text=document["text"],
                    source=document["source"],
                    page=document["page"],
                    score=score,
                    metadata=document["metadata"],
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class EvaluationStore:
    def __init__(self, path: str = DEFAULT_EVAL_PATH):
        self.path = path

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)
