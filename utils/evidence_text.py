import re


_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{80,}={0,2}")
_NON_WORD_HEAVY_RE = re.compile(r"[^\w\s.,;:!?()'-]")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "by", "with",
    "is", "are", "was", "were", "be", "as", "at", "that", "this", "it", "from",
}


def clean_evidence_text(text: str) -> str:
    if not text:
        return ""

    cleaned = _MARKDOWN_IMAGE_RE.sub(" ", text)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _BASE64_RE.sub(" ", cleaned)
    cleaned = _NON_WORD_HEAVY_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    return cleaned


def _tokenize(text: str) -> set[str]:
    parts = re.findall(r"[A-Za-z0-9']+", (text or "").lower())
    return {p for p in parts if p and p not in _STOPWORDS and len(p) > 2}


def _overlap_score(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def extract_best_excerpt(claim_text: str, raw_text: str, max_chars: int = 320) -> str:
    cleaned = clean_evidence_text(raw_text)
    if not cleaned:
        return ""

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[:max_chars]

    if not claim_text.strip():
        best = max(sentences, key=len)
        return best[:max_chars]

    scored = [(s, _overlap_score(claim_text, s)) for s in sentences]
    best_sentence, best_score = max(scored, key=lambda x: x[1])

    # If sentence alignment is weak, fall back to a compact cleaned chunk slice.
    if best_score < 0.05:
        return cleaned[:max_chars]

    if len(best_sentence) <= max_chars:
        return best_sentence
    return best_sentence[:max_chars]
