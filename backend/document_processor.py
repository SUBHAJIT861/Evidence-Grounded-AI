from pypdf import PdfReader
from pathlib import Path


# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        pages.append({
            "page": page_number,
            "text": text
        })

    return pages


# Step 2: Split text into smaller chunks
def split_into_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


# Step 3: Test PDF extraction and chunking
if __name__ == "__main__":

    pdf_path = Path("data/uploads/ACKNOWLEDGEMENT_21[1].pdf")

    if not pdf_path.exists():

        print("Error: PDF file not found!")
        print("Please check the PDF file location and name.")

    else:

        print("PDF found successfully!")

        pages = extract_text_from_pdf(pdf_path)

        print("\nTotal Pages:", len(pages))

        total_chunks = 0

        for page in pages:

            print("\n------------------------------")
            print("Page:", page["page"])

            if page["text"].strip():

                print("\nExtracted Text:")
                print(page["text"][:500])

                chunks = split_into_chunks(page["text"])

                total_chunks += len(chunks)

                print("\nTotal Chunks on this page:", len(chunks))

            else:

                print("No extractable text found on this page.")

        print("\n==============================")
        print("PDF Processing Completed!")
        print("Total Pages:", len(pages))
        print("Total Chunks:", total_chunks)
