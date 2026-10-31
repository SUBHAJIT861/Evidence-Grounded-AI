import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from pypdf import PdfReader


DOWNLOAD_FOLDER = Path("data/uploads/online_pdfs")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def download_online_pdf(url):

    parsed_url = urlparse(url)

    if parsed_url.scheme != "https":
        raise ValueError("Only HTTPS PDF URLs are allowed.")

    if not parsed_url.hostname:
        raise ValueError("Invalid PDF URL.")

    DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    file_hash = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]

    pdf_path = DOWNLOAD_FOLDER / f"{file_hash}.pdf"

    if pdf_path.exists():
        return pdf_path

    response = requests.get(
        url,
        stream=True,
        timeout=30,
        headers={
            "User-Agent": "EvidenceGroundedAI/1.0"
        }
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "application/pdf" not in content_type:
        response.close()
        raise ValueError("URL did not return a PDF file.")

    downloaded_size = 0
    first_chunk = True

    try:
        with open(pdf_path, "wb") as file:

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if not chunk:
                    continue

                if first_chunk:
                    if not chunk.startswith(b"%PDF-"):
                        raise ValueError(
                            "Downloaded file is not a valid PDF."
                        )
                    first_chunk = False

                downloaded_size += len(chunk)

                if downloaded_size > MAX_FILE_SIZE:
                    raise ValueError(
                        "PDF exceeds the 20 MB download limit."
                    )

                file.write(chunk)

        if first_chunk:
            raise ValueError("Empty PDF file.")

    except Exception:
        if pdf_path.exists():
            pdf_path.unlink()
        raise

    finally:
        response.close()

    return pdf_path


def extract_online_pdf_text(pdf_path, original_url):

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if text.strip():

            pages.append({
                "page": page_number,
                "title": pdf_path.name,
                "url": original_url,
                "text": text,
                "source_type": "online_pdf"
            })

    return pages


if __name__ == "__main__":

    pdf_url = input("Enter PDF URL: ").strip()

    try:

        print("\nDownloading PDF...")

        pdf_path = download_online_pdf(pdf_url)

        print("PDF Downloaded Successfully!")
        print("Saved Location:", pdf_path)

        print("\nExtracting PDF Text...")

        pages = extract_online_pdf_text(
            pdf_path,
            pdf_url
        )

        print("\nPDF Extraction Completed!")
        print("Total Pages with Text:", len(pages))

        for page in pages[:3]:

            print("\n------------------------------")
            print("Page:", page["page"])
            print("Text:", page["text"][:500])

    except Exception as error:

        print("\nPDF Processing Error:", error)