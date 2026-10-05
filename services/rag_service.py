import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENT_DIR = PROJECT_ROOT / "documents"


def _load_document_files(doc_dir):
    folder = Path(doc_dir)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )


def _chunk_text(text, chunk_size=160, overlap=30):
    words = re.findall(r"\S+", text)
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _document_chunks(doc_dir):
    chunks = []
    for path in _load_document_files(doc_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        for index, content in enumerate(_chunk_text(text)):
            chunks.append({
                "source": path.name,
                "chunk": index + 1,
                "content": content,
            })
    return chunks


def has_local_references(doc_dir=DEFAULT_DOCUMENT_DIR):
    return bool(_load_document_files(doc_dir))


def retrieve_documents(question, doc_dir=DEFAULT_DOCUMENT_DIR, max_results=3):
    """Retrieve the most relevant local document chunks using TF-IDF cosine similarity."""
    if not question or not question.strip() or max_results < 1:
        return []

    chunks = _document_chunks(doc_dir)
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [question.strip()])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked_indices = np.argsort(scores)[::-1]
    results = []
    for index in ranked_indices:
        if scores[index] <= 0:
            break
        chunk = chunks[index]
        results.append({
            "source": chunk["source"],
            "chunk": chunk["chunk"],
            "score": float(scores[index]),
            "snippet": chunk["content"],
        })
        if len(results) == max_results:
            break
    return results


def retrieval_fallback_answer(results):
    """Describe retrieved passages without presenting them as generated advice."""
    if not results:
        return (
            "No relevant content was found in the local reference documents. "
            "This assistant does not access private GST, banking, or government databases."
        )
    return (
        "No language model was available. The retrieved document information is shown "
        "separately below and is not a generated explanation or advice."
    )
