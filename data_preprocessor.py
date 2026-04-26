"""
data_preprocessor.py

User Story 7: Clean and preprocess data before ingestion
so retrieval results are accurate and relevant.
"""

import re
import unicodedata
from typing import List, Optional


class DataPreprocessor:
    """Cleans and preprocesses raw text data before ingestion into the RAG knowledge base."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def preprocess(self, raw_text: str, source: Optional[str] = None) -> List[dict]:
        if not raw_text or not raw_text.strip():
            return []

        print("\n" + "=" * 70)
        print(f"[PREPROCESSOR] Source: {source or 'unknown'}")
        print("=" * 70)
        print("BEFORE preprocessing (first 500 chars of raw input):")
        print("-" * 70)
        print(raw_text[:500])
        print("-" * 70)
        print(f"Raw input length: {len(raw_text)} characters\n")

        text = self.clean_text(raw_text)
        text = self.normalize_text(text)
        chunks = self.chunk_text(text)
        chunks = self.deduplicate_chunks(chunks)

        print("AFTER preprocessing (first 500 chars of cleaned text):")
        print("-" * 70)
        print(text[:500])
        print("-" * 70)
        print(f"Cleaned text length: {len(text)} characters")
        print(f"Number of chunks produced: {len(chunks)}")
        print("=" * 70 + "\n")

        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": source or "unknown",
                "chunk_index": i,
            })

        return documents

    def clean_text(self, text: str) -> str:
        """Remove HTML tags, URLs, excessive whitespace, and control characters."""
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", " ", text)

        # Remove control characters (keep newlines and tabs for structure)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
        )

        # Collapse multiple whitespace into single spaces
        text = re.sub(r"[^\S\n]+", " ", text)

        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize unicode, fix encoding artifacts, and standardize formatting."""
        # Unicode normalization (NFC form)
        text = unicodedata.normalize("NFC", text)

        # Fix common encoding artifacts
        replacements = {
            "\u2019": "'",   # right single quote
            "\u2018": "'",   # left single quote
            "\u201c": '"',   # left double quote
            "\u201d": '"',   # right double quote
            "\u2013": "-",   # en dash
            "\u2014": "-",   # em dash
            "\u2026": "...", # ellipsis
            "\u00a0": " ",   # non-breaking space
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Lowercase for consistent retrieval (optional — remove if case matters)
        # text = text.lower()

        return text

    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks, preferring to break at sentence
        or paragraph boundaries for cleaner retrieval.
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If adding this paragraph exceeds chunk_size, save current and start new
            if len(current_chunk) + len(para) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from the end of the previous chunk
                overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                current_chunk = overlap_text + " " + para
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip()

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle case where a single paragraph exceeds chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size * 1.5:
                # Force split long chunks at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", chunk)
                sub_chunk = ""
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) + 1 > chunk_size and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                    else:
                        sub_chunk = (sub_chunk + " " + sentence).strip()
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks

    def deduplicate_chunks(self, chunks: List[str]) -> List[str]:
        """Remove exact-duplicate chunks while preserving order."""
        seen = set()
        unique = []
        for chunk in chunks:
            normalized = chunk.strip().lower()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(chunk)
        return unique

    def validate_chunk(self, chunk: str, min_length: int = 20) -> bool:
        """Check if a chunk is meaningful enough to ingest."""
        if len(chunk.strip()) < min_length:
            return False
        # Reject chunks that are mostly non-alphabetic (junk data)
        alpha_ratio = sum(c.isalpha() for c in chunk) / max(len(chunk), 1)
        if alpha_ratio < 0.3:
            return False
        return True

    def preprocess_batch(self, documents: List[dict]) -> List[dict]:
        """
        Preprocess multiple raw documents at once.

        Args:
            documents: List of dicts with at least 'text' and optionally 'source'.

        Returns:
            Flat list of preprocessed, chunked document dicts.
        """
        all_processed = []
        for doc in documents:
            raw = doc.get("text", "")
            source = doc.get("source", "unknown")
            processed = self.preprocess(raw, source=source)
            # Filter out junk chunks
            processed = [p for p in processed if self.validate_chunk(p["text"])]
            all_processed.extend(processed)
        return all_processed


# ---- Quick test / example usage ----
if __name__ == "__main__":
    sample = """
    <html><body>
    <h1>Welcome to Our FAQ</h1>
    <p>Visit us at https://example.com for more info!</p>

    <p>Q: What is RAG?</p>
    <p>A: Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with language model generation. It allows chatbots
    to answer questions grounded in a specific knowledge base rather than
    relying solely on pre-trained knowledge.</p>

    <p>Q: How does chunking work?</p>
    <p>A: Chunking splits large documents into smaller, overlapping pieces
    so that a retrieval system can find the most relevant section for a
    given query. Overlap ensures context isn\u2019t lost at boundaries.</p>
    </body></html>
    """

    preprocessor = DataPreprocessor(chunk_size=300, chunk_overlap=30)
    results = preprocessor.preprocess(sample, source="faq_page.html")

    print(f"Produced {len(results)} chunks:\n")
    for doc in results:
        print(f"--- Chunk {doc['chunk_index']} (source: {doc['source']}) ---")
        print(doc["text"])
        print()