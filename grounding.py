import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List


USE_LOCAL = os.environ.get("USE_LOCAL_CLASSIFIER", "false").lower() == "true"

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None
    USE_LOCAL = True


@dataclass
class GroundedAnswer:
    answer: str
    grounded: bool
    grounding_score: float
    unsupported_spans: List[str]


class GroundedResponseEngine:
    def __init__(self):
        self.client = None if USE_LOCAL or AsyncOpenAI is None else AsyncOpenAI()

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return ""
        lines = []
        for index, chunk in enumerate(chunks, start=1):
            page = f", page {chunk.get('page')}" if chunk.get("page") else ""
            lines.append(f"[{index}] {chunk.get('source', 'Unknown')}{page}\n{chunk.get('text', '')}")
        return "\n\n".join(lines)

    def _normalize_terms(self, text: str) -> List[str]:
        stop = {"what", "why", "how", "does", "do", "is", "are", "a", "an", "the", "to", "for", "of", "and", "make"}
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        cleaned = []
        for token in tokens:
            for suffix in ("ing", "ness", "edly", "ed", "es", "s"):
                if token.endswith(suffix) and len(token) > len(suffix) + 2:
                    token = token[: -len(suffix)]
                    break
            if token not in stop:
                cleaned.append(token)
        return cleaned

    def _sentence_overlap(self, query_terms: List[str], sentence_terms: List[str]) -> int:
        score = 0
        for q in query_terms:
            for s in sentence_terms:
                if q == s or q.startswith(s[:5]) or s.startswith(q[:5]):
                    score += 1
                    break
        return score

    def _local_answer(self, question: str, chunks: List[Dict[str, Any]]) -> GroundedAnswer:
        if not chunks:
            return GroundedAnswer(
                answer="I could not find enough support in the current dataset to answer that confidently.",
                grounded=False,
                grounding_score=0.0,
                unsupported_spans=[question],
            )

        query_terms = self._normalize_terms(question)
        best_sentence = ""
        best_chunk = chunks[0]
        best_score = -1

        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
            for sentence in sentences:
                sentence_terms = self._normalize_terms(sentence)
                overlap = self._sentence_overlap(query_terms, sentence_terms)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence.strip()
                    best_chunk = chunk

        page = f", p.{best_chunk.get('page')}" if best_chunk.get("page") else ""
        if not best_sentence:
            best_sentence = re.sub(r"\s+", " ", best_chunk.get("text", "").strip())
        answer = f"{best_sentence} [Source: {best_chunk.get('source')}{page}]"
        return GroundedAnswer(answer=answer, grounded=True, grounding_score=1.0, unsupported_spans=[])

    async def answer(self, question: str, chunks: List[Dict[str, Any]], routed_prompt: str) -> GroundedAnswer:
        print(f"DEBUG: USE_LOCAL={USE_LOCAL} client={self.client}")  # ← add this
        if USE_LOCAL or self.client is None:
            return self._local_answer(question, chunks)

        if not chunks:
            return GroundedAnswer(
                answer="I could not find enough support in the current dataset to answer that confidently.",
                grounded=False,
                grounding_score=0.0,
                unsupported_spans=[question],
            )

        context = self._build_context(chunks)
        # grounding.py - Modified logic
        system_prompt = (
            "You are a technical Quantum Mechanics assistant. "
            "Your goal is precision and brevity. "
            "1. Provide mathematical definitions ($\Delta x \Delta p \geq \hbar/2$) whenever possible. "
            "2. Avoid historical anecdotes (e.g., Schrödinger's cat, EPR) or hypothetical characters (e.g., Eve) "
            "unless explicitly asked. "
            "3. Use a 'Technical & Concise' tone."
        )


        user_prompt = (
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Draft instruction:\n{routed_prompt}\n\n"
            "Return JSON with keys: \n"
            "- answer: (string) The technical response.\n"
            "- grounded: (boolean) True if the core physics is supported by the context.\n"
            "- grounding_score: (float 0.0-1.0) Score how well the answer reflects the context. "
            "Prioritize semantic meaning and physical principles over literal string matches. "
            "Do not penalize for using synonymous technical terms (e.g., 'basis state' vs 'eigenstate').\n"
            "- unsupported_spans: (list) Any fluff or unverified claims."
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            raw = (response.choices[0].message.content or "").strip()
            payload = json.loads(raw) if raw else {}
            return GroundedAnswer(
                answer=str(payload.get("answer", "")).strip() or self._local_answer(question, chunks).answer,
                grounded=bool(payload.get("grounded", False)),
                grounding_score=float(payload.get("grounding_score", 0.0)),
                unsupported_spans=list(payload.get("unsupported_spans", [])),
            )
        except Exception as e:
            print(f"DEBUG grounding exception: {e}")
            return self._local_answer(question, chunks)
