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

def scrape_url(url: str) -> dict:
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)

        if response.status_code != 200:
            return {"content": "", "publish_date": None}

        html = response.text

        # Extract content
        extracted = trafilatura.extract(html)

        # Extract metadata
        metadata = trafilatura.extract_metadata(html)

        content = ""
        if extracted and len(extracted.strip()) > 200:
            content = clean_text(extracted)
        else:
            content = fallback_bs4(html)

        publish_date = None
        if metadata and metadata.date:
            publish_date = metadata.date

        return {
            "content": content,
            "publish_date": publish_date
        }

    except Exception as e:
        print(f"[SCRAPER ERROR] {e}")
        return {"content": "", "publish_date": None}

# Clean text function
def clean_text(text: str) -> str:
    return " ".join(text.split())  # remove extra spaces
      # limit size for LLM


# BeautifulSoup fallback
def fallback_bs4(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        return clean_text(text)

    except Exception as e:
        print(f"[BS4 FALLBACK ERROR] {e}")
        return ""