import requests
import trafilatura
from bs4 import BeautifulSoup
from config.constants.scraper_constants import MIN_CONTENT_WORDS, MAX_CONTENT_CHARS, SCRAPE_TIMEOUT

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
    text = " ".join(text.split())
    return text[:MAX_CONTENT_CHARS]


def fallback_bs4(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
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
        "publish_date": str | None,
        "error_type": str | None   
    }
    """

    try:
        response = requests.get(url, headers=HEADERS, timeout=SCRAPE_TIMEOUT)

        # -------------------------------
        # HTTP ERROR CLASSIFICATION
        # -------------------------------
        if response.status_code != 200:
            print(f"[SCRAPER WARNING] Non-200 response: {url} | status={response.status_code}")

            if response.status_code == 401 or response.status_code == 403:
                error_type = "auth_error"
            elif response.status_code == 404:
                error_type = "not_found"
            elif 500 <= response.status_code < 600:
                error_type = "server_error"
            else:
                error_type = "http_error"

            return {"content": "", "publish_date": None, "error_type": error_type}

        html = response.text

        extracted = trafilatura.extract(html)
        metadata = trafilatura.extract_metadata(html)

        if not extracted:
            print(f"[SCRAPER DEBUG] Trafilatura returned empty: {url}")

        content = ""

        if extracted:
            word_count = len(extracted.split())
            if word_count >= MIN_CONTENT_WORDS:
                content = clean_text(extracted)
            else:
                print(f"[SCRAPER DEBUG] Low word count ({word_count}) → fallback: {url}")

        if not content:
            print(f"[SCRAPER FALLBACK] Using BS4 for: {url}")
            content = fallback_bs4(html)

        # -------------------------------
        # FINAL VALIDATION
        # -------------------------------
        if not content or len(content.split()) < MIN_CONTENT_WORDS:
            print(f"[SCRAPER ERROR] No usable content: {url}")
            return {"content": "", "publish_date": None, "error_type": "content_unusable"}  

        # -------------------------------
        # METADATA
        # -------------------------------
        publish_date = None
        if metadata and getattr(metadata, "date", None):
            publish_date = metadata.date

        return {
            "content": content,
            "publish_date": publish_date,
            "error_type": None  
        }

    # -------------------------------
    #  NETWORK / TIMEOUT HANDLING
    # -------------------------------
    except requests.exceptions.Timeout:
        print(f"[SCRAPER ERROR] Timeout: {url}")
        return {"content": "", "publish_date": None, "error_type": "timeout_error"}

    except requests.exceptions.ConnectionError:
        print(f"[SCRAPER ERROR] Connection failed: {url}")
        return {"content": "", "publish_date": None, "error_type": "network_error"}

    except Exception as e:
        print(f"[SCRAPER ERROR] URL: {url} | ERROR: {e}")
        return {"content": "", "publish_date": None, "error_type": "unknown_error"}