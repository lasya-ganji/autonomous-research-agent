import requests
import trafilatura
from bs4 import BeautifulSoup

# Real browser-like headers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Content control (IMPORTANT for LLM + cost)
MAX_CONTENT_CHARS = 5000
MIN_CONTENT_WORDS = 100


def clean_text(text: str) -> str:
    """
    Normalize and trim content for safe LLM usage
    """
    if not text:
        return ""

    text = " ".join(text.split())  
    return text[:MAX_CONTENT_CHARS]


def fallback_bs4(html: str) -> str:
    """
    Fallback content extraction using BeautifulSoup
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove noisy tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return clean_text(text)

    except Exception as e:
        print(f"[BS4 FALLBACK ERROR] {e}")
        return ""


def scrape_url(url: str) -> dict:
    """
    Scrape webpage content and metadata

    Returns:
    {
        "content": str,
        "publish_date": str | None
    }
    """

    try:
        response = requests.get(url, headers=HEADERS, timeout=8)

        if response.status_code != 200:
            return {"content": "", "publish_date": None}

        html = response.text

        # -------------------------------
        # PRIMARY EXTRACTION (trafilatura)
        # -------------------------------
        extracted = trafilatura.extract(html)
        metadata = trafilatura.extract_metadata(html)

        content = ""

        if extracted:
            word_count = len(extracted.split())

            if word_count >= MIN_CONTENT_WORDS:
                content = clean_text(extracted)

        # -------------------------------
        # FALLBACK EXTRACTION
        # -------------------------------
        if not content:
            content = fallback_bs4(html)

        # -------------------------------
        # METADATA
        # -------------------------------
        publish_date = None
        if metadata and getattr(metadata, "date", None):
            publish_date = metadata.date

        return {
            "content": content,
            "publish_date": publish_date
        }

    except Exception as e:
        print(f"[SCRAPER ERROR] URL: {url} | ERROR: {e}")
        return {"content": "", "publish_date": None}