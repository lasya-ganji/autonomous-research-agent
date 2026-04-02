from datetime import datetime
import requests

from models.citation_models import Citation
from models.enums import CitationStatus


# Simple in-memory cache
_url_cache = {}


def validate_url(url: str) -> CitationStatus:

    if not url:
        return CitationStatus.broken


    # CACHE HIT

    if url in _url_cache:
        return _url_cache[url]

    try:
     
    
        resp = requests.get(
            url,
            timeout=5,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        status_code = resp.status_code

        if status_code == 200:
            status = CitationStatus.valid

        elif 300 <= status_code < 400:
            status = CitationStatus.stale

        elif status_code >= 400:
            status = CitationStatus.broken

        else:
            status = CitationStatus.stale

    except requests.Timeout:

        status = CitationStatus.stale

    except requests.RequestException:
        status = CitationStatus.stale


    # CACHE STORE

    _url_cache[url] = status

    return status


# BUILD CITATION
def build_citation(result, citation_id: str) -> Citation:

    url = str(getattr(result, "url", ""))

    return Citation(
        citation_id=citation_id,
        title=getattr(result, "title", "Untitled"),
        url=url,
        quality_score=float(getattr(result, "quality_score", 0.5) or 0.5),
        status=validate_url(url),
        date_accessed=datetime.now().isoformat()
    )