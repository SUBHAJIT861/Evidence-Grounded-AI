import os
import base64

from urllib.parse import (
    urlparse,
    parse_qs,
    unquote,
    urlunparse,
    parse_qsl,
    urlencode
)

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def clean_url(url):
    """
    Decode Bing tracking URLs and return the original destination URL.
    """

    try:
        parsed = urlparse(url)

        if (
            "bing.com" in parsed.netloc.lower()
            and parsed.path == "/aclick"
        ):

            params = parse_qs(parsed.query)
            encoded_url = params.get("u", [""])[0]

            if encoded_url:

                decoded_url = base64.urlsafe_b64decode(
                    encoded_url + "=" * (-len(encoded_url) % 4)
                ).decode("utf-8")

                decoded_url = unquote(decoded_url)

                if decoded_url.startswith(("http://", "https://")):
                    return decoded_url

        return url

    except Exception:
        return url


def normalize_url(url):
    """
    Create a canonical URL for duplicate detection.
    Ignore query parameters so tracking variants of the
    same article are treated as one source.
    """

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower()
            + "://"
            + parsed.netloc.lower()
            + parsed.path.rstrip("/")
        )

    except Exception:
        return url.lower().rstrip("/")


def web_search(query, max_results=5):

    if not query.strip():
        return []

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found in .env file."
        )

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query,
        max_results=max_results,
        search_depth="advanced"
    )

    sources = []
    seen_urls = set()

    for result in response.get("results", []):

        original_url = result.get("url", "")

        if not original_url:
            continue

        url = clean_url(original_url)

        normalized_url = normalize_url(url)

        if normalized_url in seen_urls:
            continue

        seen_urls.add(normalized_url)

        sources.append({
            "title": result.get("title", ""),
            "url": url,
            "content": result.get("content", "")
        })

    return sources