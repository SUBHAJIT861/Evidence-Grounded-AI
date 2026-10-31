import faiss
import numpy as np

from backend.embeddings import (
    load_embedding_model,
    create_embeddings
)


# Step 1: Create FAISS index
def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    return index


# Step 2: Search relevant chunks
def search_relevant_chunks(
    question,
    model,
    index,
    chunks,
    top_k=3
):

    question_embedding = create_embeddings(
        [question],
        model
    )

    distances, indices = index.search(
        question_embedding.astype("float32"),
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "similarity_score": float(score)
        })

    return results


# Step 3: Test retrieval
if __name__ == "__main__":

    from pathlib import Path

    from backend.document_processor import (
        extract_text_from_pdf,
        split_into_chunks
    )

    pdf_path = Path("data/uploads/ACKNOWLEDGEMENT_21[1].pdf")

    if not pdf_path.exists():

        print("Error: PDF not found!")

    else:

        pages = extract_text_from_pdf(pdf_path)

        chunks = []

        for page in pages:

            page_chunks = split_into_chunks(page["text"])

            for chunk in page_chunks:

                if chunk.strip():

                    chunks.append({
                        "page": page["page"],
                        "text": chunk
                    })

        if not chunks:

            print("No text available!")

        else:

            print("Loading embedding model...")

            model = load_embedding_model()

            texts = [chunk["text"] for chunk in chunks]

            print("Creating embeddings...")

            embeddings = create_embeddings(texts, model)

            print("Building FAISS index...")

            index = create_faiss_index(embeddings)

            question = input("\nEnter your question about the PDF: ")

            results = search_relevant_chunks(
                question,
                model,
                index,
                chunks,
                top_k=3
            )

            print("\nRelevant Evidence:")

            for i, result in enumerate(results, start=1):

                print("\n------------------------------")
                print("Result:", i)
                print("PDF Page:", result["page"])
                print("Similarity Score:", round(result["similarity_score"], 4))
                print("Text:", result["text"])

            print("\nRetrieval Completed!")