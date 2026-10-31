import requests
import re

from bs4 import BeautifulSoup
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    )
}

TIMEOUT = 20


def extract_webpage_content(url):

    if not url:
        return None

    parsed = urlparse(url)

    if parsed.scheme != "https" or not parsed.hostname:
        return None

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=TIMEOUT
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary HTML elements
        for element in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "button",
            "noscript",
            "svg"
        ]):
            element.decompose()

        # Prefer the main article content
        article = soup.find("article")

        if article is None:
            article = soup.find("main")

        if article is None:
            article = soup.body

        if article is None:
            return None

        paragraphs = article.find_all(
            ["p", "h1", "h2", "h3", "li"]
        )

        extracted_text = []

        for element in paragraphs:

            text = element.get_text(
                separator=" ",
                strip=True
            )

            text = re.sub(r"\s+", " ", text)

            if len(text.split()) < 6:
                continue

            extracted_text.append(text)

        content = "\n".join(extracted_text)

        if len(content.strip()) < 100:
            return None

        return {
            "title": (
                soup.title.get_text(strip=True)
                if soup.title else urlparse(response.url).netloc
            ),
            "url": response.url,
            "content": content,
            "source_type": "web"
        }

    except requests.RequestException as error:
        print("Webpage request failed:", error)
        return None

    except Exception as error:
        print("Webpage extraction failed:", error)
        return None

    
if __name__ == "__main__":

    test_url = (
        "https://cloud.google.com/learn/"
        "what-is-artificial-intelligence"
    )

    result = extract_webpage_content(test_url)

    if result:
        print("\nTitle:", result["title"])
        print("\nURL:", result["url"])
        print("\nExtracted Content:")
        print(result["content"][:2000])
        print("\nExtraction Successful!")
    else:
        print("Could not extract webpage content.")

        