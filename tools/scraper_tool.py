import requests
import trafilatura
from bs4 import BeautifulSoup

from config.constants.scraper_constants import (
    MIN_CONTENT_WORDS,
    MAX_CONTENT_CHARS,
    SCRAPE_TIMEOUT,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split())[:MAX_CONTENT_CHARS]


def _bs4_extract(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        return clean_text(soup.get_text(separator=" "))
    except Exception:
        return ""


def scrape_url(url: str) -> dict:
    """
    Fetches and extracts article text from a URL.

    Stages:
      1. HTTP GET — classify HTTP errors.
      2. Content-Type check — reject non-HTML responses (PDFs, media, etc.).
      3. Trafilatura (favor_precision=True) — reject non-article pages (None result).
      4. BS4 fallback — only when Trafilatura returns thin content, not None.

    Returns:
        {
            "status":       "success" | "failed" | "low_content",
            "content":      str,
            "publish_date": str | None,
            "error_type":   str | None,
        }
    """

    def _failed(error_type: str) -> dict:
        return {"status": "failed", "content": "", "publish_date": None, "error_type": error_type}

    def _low_content() -> dict:
        return {"status": "low_content", "content": "", "publish_date": None, "error_type": "content_unusable"}

    try:
        response = requests.get(url, headers=HEADERS, timeout=SCRAPE_TIMEOUT)

        if response.status_code in (401, 403):
            print(f"[SCRAPER ERROR] url={url} status={response.status_code} reason=blocked")
            return _failed("auth_blocked")

        if response.status_code == 404:
            print(f"[SCRAPER ERROR] url={url} status=404 reason=not_found")
            return _failed("not_found")

        if 500 <= response.status_code < 600:
            print(f"[SCRAPER ERROR] url={url} status={response.status_code} reason=server_error")
            return _failed("server_error")

        if response.status_code != 200:
            print(f"[SCRAPER ERROR] url={url} status={response.status_code} reason=http_error")
            return _failed("http_error")

        # Gate: reject non-HTML responses (PDFs, images, media) before parsing
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            print(f"[SCRAPER ERROR] url={url} content_type='{content_type}' reason=non_html_content")
            return _failed("non_html_content")

        html = response.text

        # favor_precision=True makes Trafilatura treat non-article pages as None
        extracted = trafilatura.extract(html, favor_precision=True)
        metadata = trafilatura.extract_metadata(html)

        if extracted is None:
            print(f"[SCRAPER ERROR] url={url} reason=non_article (trafilatura precision reject)")
            return _failed("non_article")

        if len(extracted.split()) >= MIN_CONTENT_WORDS:
            content = clean_text(extracted)
        else:
            # Trafilatura found partial content — try BS4 before giving up
            print(f"[SCRAPER] url={url} trafilatura_thin={len(extracted.split())} words — BS4 fallback")
            content = _bs4_extract(html)
            if not content or len(content.split()) < MIN_CONTENT_WORDS:
                print(f"[QUALITY] Rejected low content url={url} word_count={len(content.split()) if content else 0}")
                return _low_content()
            content = clean_text(content)

        publish_date = None
        if metadata and getattr(metadata, "date", None):
            publish_date = metadata.date

        return {
            "status": "success",
            "content": content,
            "publish_date": publish_date,
            "error_type": None,
        }

    except requests.exceptions.Timeout:
        print(f"[SCRAPER ERROR] url={url} reason=timeout")
        return _failed("timeout_error")

    except requests.exceptions.ConnectionError:
        print(f"[SCRAPER ERROR] url={url} reason=connection_failed")
        return _failed("network_error")

    except Exception as e:
        print(f"[SCRAPER ERROR] url={url} reason={e}")
        return _failed("unknown_error")
