import requests

from models.enums import CitationStatus
from config.constants.node_constants.citation_constants import CITATION_VALIDATE_TIMEOUT

# Process-level cache. Avoids re-validating the same URL across steps / supervisor loops.
_url_cache: dict[str, CitationStatus] = {}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _classify_status(code: int) -> CitationStatus:
    if code == 200:
        return CitationStatus.valid
    if 300 <= code < 400:
        return CitationStatus.stale
    if code >= 400:
        return CitationStatus.broken
    return CitationStatus.stale


def validate_url(url: str) -> CitationStatus:
    """
    Lightweight reachability check for citation status.

    Uses a streaming GET so the server processes a real request (accurate status
    codes) but the response body is never downloaded — the connection is closed
    as soon as the status line and headers arrive.

    HEAD was considered but rejected: many WAFs, CDNs, and web frameworks
    mishandle HEAD by returning 403/405 even when the page is fully accessible,
    or return 200 for HEAD while serving a redirect body on GET. Streaming GET
    gives the same network cost with reliable status codes.

    Cached per-process: multi-step supervisor loops don't re-probe the same URL.
    """
    if not url:
        return CitationStatus.broken

    if url in _url_cache:
        return _url_cache[url]

    try:
        with requests.get(
            url,
            timeout=CITATION_VALIDATE_TIMEOUT,
            allow_redirects=True,
            headers=_HEADERS,
            stream=True,
        ) as resp:
            status = _classify_status(resp.status_code)

    except requests.Timeout:
        status = CitationStatus.stale

    except requests.RequestException:
        status = CitationStatus.broken

    _url_cache[url] = status
    return status


def status_from_scrape(error_type: str | None) -> CitationStatus:
    """
    Maps a scrape_url failure error_type to the appropriate citation status.
    Used when scrape_url has already hit the URL, so a separate HTTP probe
    would be wasteful.
    """
    if error_type is None:
        return CitationStatus.valid
    if error_type in {"not_found", "http_error"}:
        return CitationStatus.broken
    # auth_blocked, timeout_error, network_error, non_article, non_html_content,
    # server_error, content_unusable, unknown_error — URL exists but content not usable
    return CitationStatus.stale
