from sentence_transformers import SentenceTransformer
from pathlib import Path

from backend.document_processor import (
    extract_text_from_pdf,
    split_into_chunks
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(texts, model):
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


if __name__ == "__main__":

    pdf_path = Path("data/uploads/ACKNOWLEDGEMENT_21[1].pdf")

    if not pdf_path.exists():
        print("Error: PDF file not found!")

    else:
        print("Loading PDF...")

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
            print("No text available for embedding.")

        else:
            print("Loading embedding model...")

            model = load_embedding_model()

            texts = [chunk["text"] for chunk in chunks]

            embeddings = create_embeddings(texts, model)

            print("\nEmbedding Completed!")
            print("Total chunks:", len(chunks))
            print("Embedding shape:", embeddings.shape)
            print("First chunk page:", chunks[0]["page"])
            print("First vector sample:", embeddings[0][:5])