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
    print("UDYAMPULSE AI - RAG COMPLIANCE ASSISTANT")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved documents:")
    for item in retrieved:
        print(f"- {item['source']} (score={item['score']})")

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources used:")
    for source in response["sources"]:
        print(f"- {source}")

    print("\nExplanation")
    print("-" * 40)
    print(
        "This is a beginner-friendly RAG prototype. The assistant retrieves local public-reference documents, "
        "extracts relevant snippets, and then answers using those snippets as context."
    )


if __name__ == "__main__":
    main()
