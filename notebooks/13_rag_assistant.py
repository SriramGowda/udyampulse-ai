"""UdyamPulse AI | Phase 13: demonstrate local document-grounded RAG.

Inputs: documents/*.txt and documents/*.md via production RAG services
Outputs: ranked local excerpts, source filenames, and grounded/fallback response
Method: reuse TF-IDF retrieval and Gemini/Ollama/retrieval-only orchestration.
"""

import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from services.ai_service import generate_rag_answer, retrieve_documents


# --------------------------------------------------
# Phase 5: RAG Compliance Assistant
# --------------------------------------------------


def main():
    question = (
        "How should an MSME monitor cash flow and working capital to avoid financial stress?"
    )

    retrieved = retrieve_documents(question, doc_dir="documents", max_results=3)
    response = generate_rag_answer(question, doc_dir="documents")

    print("=" * 70)
    print("UDYAMPULSE AI - PHASE 13: LOCAL DOCUMENT-GROUNDED RAG")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved documents:")
    for item in retrieved:
        print(
            f"- {item['source']} (chunk {item['chunk']}, "
            f"score={item['score']:.3f})"
        )
        print(f"  Excerpt: {item['snippet']}")

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources used:")
    for source in response["sources"]:
        print(f"- {source}")

    print("\nExplanation")
    print("-" * 40)
    print(
        "This is local document-grounded RAG: TXT/MD files are chunked and ranked "
        "with TF-IDF/cosine similarity, then passed to the production Gemini/Ollama "
        "service. If both are unavailable, retrieved excerpts are returned without "
        "generated advice. It is not internet-scale search or legal guidance."
    )


if __name__ == "__main__":
    main()
