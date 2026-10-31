from pathlib import Path

from backend.document_processor import (
    extract_text_from_pdf,
    split_into_chunks
)

from backend.embeddings import (
    load_embedding_model,
    create_embeddings
)

from backend.retrieval import (
    create_faiss_index,
    search_relevant_chunks
)

from backend.answer_generator import generate_answer
from backend.verifier import verify_claim
from backend.claim_extractor import extract_claims
from backend.search import web_search

from backend.webpage_processor import (
    extract_webpage_content
)

from backend.online_pdf_retrieval import (
    retrieve_online_pdf_evidence
)


def process_uploaded_pdf(pdf_path, question, model=None):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError("Uploaded PDF not found.")

    pages = extract_text_from_pdf(pdf_path)
    chunks = []

    for page in pages:
        page_chunks = split_into_chunks(
            page["text"],
            chunk_size=500
        )

        for chunk in page_chunks:
            if chunk.strip():
                chunks.append({
                    "page": page["page"],
                    "title": pdf_path.name,
                    "url": "",
                    "text": chunk,
                    "source_type": "local_pdf"
                })

    if not chunks:
        raise ValueError("No readable text found in uploaded PDF.")

    if model is None:
        model = load_embedding_model()

    embeddings = create_embeddings(
        [item["text"] for item in chunks],
        model
    )

    index = create_faiss_index(embeddings)

    evidence = search_relevant_chunks(
        question,
        model,
        index,
        chunks,
        top_k=3
    )

    for item in evidence:
        item["title"] = pdf_path.name
        item["url"] = ""
        item["source_type"] = "local_pdf"

    return evidence, model


def retrieve_web_evidence(question, model=None):
    web_evidence = []

    try:
        web_results = web_search(
            question,
            max_results=5
        )

        for result in web_results:
            url = result.get("url", "")
            title = result.get("title", "Unknown")

            try:
                webpage = extract_webpage_content(url)
            except Exception:
                webpage = None

            if webpage:
                webpage_chunks = split_into_chunks(
                    webpage["content"],
                    chunk_size=300
                )

                if not webpage_chunks:
                    continue

                if model is None:
                    model = load_embedding_model()

                webpage_embeddings = create_embeddings(
                    webpage_chunks,
                    model
                )

                question_embedding = create_embeddings(
                    [question],
                    model
                )

                scores = (
                    webpage_embeddings @ question_embedding[0]
                )

                top_indices = scores.argsort()[::-1][:2]

                for idx in top_indices:
                    web_evidence.append({
                        "page": "Web",
                        "title": webpage["title"],
                        "url": webpage["url"],
                        "text": webpage_chunks[idx],
                        "source_type": "web",
                        "similarity_score": float(scores[idx])
                    })

            else:
                snippet = result.get("content", "")

                if snippet.strip():
                    web_evidence.append({
                        "page": "Search Snippet",
                        "title": title,
                        "url": url,
                        "text": snippet,
                        "source_type": "web_snippet",
                        "similarity_score": 0.0
                    })

    except Exception as error:
        print("Web Search Error:", error)

    return web_evidence, model


def run_pipeline(question, uploaded_pdf_path=None, search_mode="general"):
    """
    Run the complete evidence-grounded QA pipeline.

    search_mode:
        general  = Web + Online PDFs
        uploaded = Uploaded PDF only
        combined = Uploaded PDF + Web + Online PDFs
    """

    question = (question or "").strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if search_mode not in ("general", "uploaded", "combined"):
        raise ValueError("Invalid search mode.")

    if search_mode in ("uploaded", "combined") and not uploaded_pdf_path:
        raise ValueError("Please upload a PDF for this search mode.")

    pdf_evidence = []
    web_evidence = []
    online_pdf_evidence = []
    errors = []

    model = None

    # 1. Uploaded PDF
    if search_mode in ("uploaded", "combined"):
        try:
            pdf_evidence, model = process_uploaded_pdf(
                uploaded_pdf_path,
                question,
                model
            )
        except Exception as error:
            errors.append(f"Uploaded PDF: {error}")

    # 2. Web Search
    if search_mode in ("general", "combined"):
        web_evidence, model = retrieve_web_evidence(
            question,
            model
        )

    # 3. Online PDF Search
    if search_mode in ("general", "combined"):
        try:
            online_pdf_evidence = retrieve_online_pdf_evidence(
                question,
                max_pdf_results=2,
                top_k=5
            )
        except Exception as error:
            errors.append(f"Online PDF Search: {error}")

    # 4. Combine evidence
    evidence = (
        pdf_evidence
        + web_evidence
        + online_pdf_evidence
    )

    evidence_summary = {
        "uploaded_pdf": len(pdf_evidence),
        "webpages": sum(
            1 for item in web_evidence
            if item.get("source_type") == "web"
        ),
        "web_snippets": sum(
            1 for item in web_evidence
            if item.get("source_type") == "web_snippet"
        ),
        "online_pdfs": len(online_pdf_evidence),
        "total": len(evidence)
    }

    if not evidence:
        return {
            "question": question,
            "answer": "No evidence was retrieved. Unable to generate an evidence-based answer.",
            "sources": [],
            "evidence_summary": evidence_summary,
            "claims": [],
            "verification_results": [],
            "status_counts": {},
            "errors": errors
        }

    # 5. Generate answer
    answer_result = generate_answer(
        question,
        evidence
    )

    answer = answer_result.get("answer", "")
    sources = answer_result.get("sources", [])

    if not answer or not answer.strip():
        answer = "No answer could be generated from the retrieved evidence."

    # 6. Extract claims
    claims = extract_claims(
        answer,
        max_claims=10
    )

    # 7. Verify claims
    verification_results = []

    for claim in claims:
        try:
            verification = verify_claim(
                claim,
                evidence
            )

            verification_results.append(verification)

        except Exception as error:
            verification_results.append({
                "claim": claim,
                "status": "Unverified",
                "reason": str(error),
                "confidence": 0.0,
                "evidence": "Verification failed.",
                "page": "",
                "title": "",
                "url": ""
            })

    # 8. Count verification statuses
    status_counts = {
        "Supported": 0,
        "Unsupported": 0,
        "Contradicted": 0,
        "Partially Supported": 0,
        "Unverified": 0
    }

    for result in verification_results:
        status = result.get("status", "Unverified")

        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["Unverified"] += 1

    # 9. Return everything to the frontend
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "evidence_summary": evidence_summary,
        "claims": claims,
        "verification_results": verification_results,
        "status_counts": status_counts,
        "errors": errors
    }


def main(question=None, uploaded_pdf_path=None):
    """Command-line interface for the same pipeline."""

    print("===================================")
    print("Evidence-Grounded AI System")
    print("===================================")

    if question is None:
        question = input("\nEnter your question: ").strip()

    try:
        result = run_pipeline(
            question=question,
            uploaded_pdf_path=uploaded_pdf_path,
            search_mode="combined" if uploaded_pdf_path else "general"
        )
    except Exception as error:
        print("Pipeline Error:", error)
        return

    print("\nANSWER")
    print(result["answer"])

    print("\nEVIDENCE SUMMARY")
    for name, count in result["evidence_summary"].items():
        print(f"{name}: {count}")

    print("\nSOURCES")
    for source in result["sources"]:
        print("Title:", source.get("title", ""))
        print("Page:", source.get("page", ""))
        print("URL:", source.get("url", ""))
        print("Type:", source.get("source_type", ""))
        print("-------------------------")

    print("\nVERIFICATION REPORT")
    for item in result["verification_results"]:
        print("Claim:", item.get("claim", ""))
        print("Status:", item.get("status", ""))
        print("Confidence:", item.get("confidence", 0))
        print("Reason:", item.get("reason", ""))
        print("-------------------------")

    print("\nSTATUS COUNTS")
    for status, count in result["status_counts"].items():
        print(f"{status}: {count}")

    if result["errors"]:
        print("\nWARNINGS")
        for error in result["errors"]:
            print(error)

    print("\nPIPELINE COMPLETED!")


if __name__ == "__main__":
    main()