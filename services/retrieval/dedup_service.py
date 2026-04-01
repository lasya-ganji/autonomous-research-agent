from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    try:
        url = str(url)
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except:
        return str(url)


def deduplicate_results(raw_results, seen_urls: set):
    unique_results = []

    for r in raw_results:
        url = str(getattr(r, "url", ""))
        if not url:
            continue

        norm_url = normalize_url(url)

        if norm_url in seen_urls:
            continue

        seen_urls.add(norm_url)

        unique_results.append((r, norm_url))

    return unique_results