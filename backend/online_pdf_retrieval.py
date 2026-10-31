from backend.online_pdf_search import search_online_pdfs
from backend.online_pdf_processor import (
    download_online_pdf,
    extract_online_pdf_text
)
from backend.document_processor import split_into_chunks
from backend.embeddings import (
    load_embedding_model,
    create_embeddings
)
from backend.retrieval import create_faiss_index


def retrieve_online_pdf_evidence(
    question,
    max_pdf_results=2,
    top_k=5
):
    if not question.strip():
        return []

    print("\nSearching for online PDFs...")

    pdf_sources = search_online_pdfs(
        question,
        max_results=5
    )

    if not pdf_sources:
        print("No online PDFs found.")
        return []

    all_chunks = []

    for pdf in pdf_sources[:max_pdf_results]:
        try:
            print("\nProcessing:", pdf.get("title", "Unknown PDF"))

            pdf_path = download_online_pdf(pdf["url"])

            pages = extract_online_pdf_text(
                pdf_path,
                pdf["url"]
            )

            for page in pages:
                chunks = split_into_chunks(
                    page["text"],
                    chunk_size=500
                )

                for chunk in chunks:
                    if chunk.strip():
                        all_chunks.append({
                            "title": pdf.get("title", "Online PDF"),
                            "url": pdf["url"],
                            "page": page["page"],
                            "text": chunk,
                            "source_type": "online_pdf"
                        })

        except Exception as error:
            print("PDF processing skipped:", error)
            continue

    if not all_chunks:
        print("No readable PDF evidence found.")
        return []

    print("\nCreating PDF embeddings...")

    model = load_embedding_model()

    embeddings = create_embeddings(
        [item["text"] for item in all_chunks],
        model
    )

    index = create_faiss_index(embeddings)

    question_embedding = create_embeddings(
        [question],
        model
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        min(top_k, len(all_chunks))
    )

    evidence = []

    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        item = all_chunks[idx].copy()
        item["similarity_score"] = float(score)
        evidence.append(item)

    return evidence


if __name__ == "__main__":
    question = input("Enter your question: ").strip()

    try:
        results = retrieve_online_pdf_evidence(question)

        print("\nOnline PDF Evidence Results")
        print("Total evidence chunks:", len(results))

        for i, item in enumerate(results, start=1):
            print("\n------------------------------")
            print("Evidence:", i)
            print("Title:", item["title"])
            print("Page:", item["page"])
            print("URL:", item["url"])
            print("Similarity:", round(item["similarity_score"], 4))
            print("Text:", item["text"][:500])

    except Exception as error:
        print("\nRetrieval Error:", error)