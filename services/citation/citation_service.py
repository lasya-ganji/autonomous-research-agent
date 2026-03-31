from datetime import datetime
import requests
from models.citation_models import Citation
from models.enums import CitationStatus


def validate_url(url: str) -> CitationStatus:
    try:
        resp = requests.head(url, timeout=3, allow_redirects=True)

        if resp.status_code == 200:
            return CitationStatus.valid
        elif resp.status_code in (301, 302):
            return CitationStatus.stale
        else:
            return CitationStatus.broken

    except requests.RequestException:
        return CitationStatus.broken


def build_citation(result, citation_id: str) -> Citation:
    return Citation(
        citation_id=citation_id,
        title=getattr(result, "title", "Untitled"),
        url=getattr(result, "url", ""),
        quality_score=getattr(result, "quality_score", 0.5),
        status=validate_url(getattr(result, "url", "")),
        date_accessed=datetime.now().isoformat()
    )