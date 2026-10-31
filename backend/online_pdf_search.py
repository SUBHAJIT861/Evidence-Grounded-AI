from backend.search import web_search


def search_online_pdfs(question, max_results=5):

    if not question.strip():
        return []

    search_query = f"{question} filetype:pdf"

    results = web_search(
        search_query,
        max_results=max_results
    )

    pdf_sources = []

    for result in results:

        url = result.get("url", "")

        if not url:
            continue

        if ".pdf" in url.lower():

            pdf_sources.append({
                "title": result.get("title", ""),
                "url": url,
                "content": result.get("content", ""),
                "source_type": "online_pdf"
            })

    return pdf_sources


if __name__ == "__main__":

    question = input("Enter your question: ").strip()

    try:
        results = search_online_pdfs(question)

        print("\nOnline PDF Search Results:")
        print("Total PDFs found:", len(results))

        for i, pdf in enumerate(results, start=1):

            print("\n------------------------------")
            print("PDF:", i)
            print("Title:", pdf["title"])
            print("URL:", pdf["url"])
            print("Content:", pdf["content"][:300])

        print("\nOnline PDF Search Completed!")

    except Exception as error:
        print("Search Error:", error)