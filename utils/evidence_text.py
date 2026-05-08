import re

from config.constants.evidence_text_constants import (
    BASE64_PATTERN,
    DEFAULT_EVIDENCE_EXCERPT_MAX_CHARS,
    EVIDENCE_STOPWORDS,
    MARKDOWN_IMAGE_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MIN_TOKEN_LENGTH,
    NON_WORD_HEAVY_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    URL_PATTERN,
    WEAK_SENTENCE_ALIGNMENT_THRESHOLD,
    WHITESPACE_PATTERN,
)

_URL_RE = re.compile(URL_PATTERN, re.IGNORECASE)
_MARKDOWN_IMAGE_RE = re.compile(MARKDOWN_IMAGE_PATTERN)
_MARKDOWN_LINK_RE = re.compile(MARKDOWN_LINK_PATTERN)
_BASE64_RE = re.compile(BASE64_PATTERN)
_NON_WORD_HEAVY_RE = re.compile(NON_WORD_HEAVY_PATTERN)
_WHITESPACE_RE = re.compile(WHITESPACE_PATTERN)
_SENTENCE_SPLIT_RE = re.compile(SENTENCE_SPLIT_PATTERN)


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
    return {
        p for p in parts
        if p and p not in EVIDENCE_STOPWORDS and len(p) >= MIN_TOKEN_LENGTH
    }


def _overlap_score(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def extract_best_excerpt(
    claim_text: str,
    raw_text: str,
    max_chars: int = DEFAULT_EVIDENCE_EXCERPT_MAX_CHARS,
) -> str:
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

    if best_score < WEAK_SENTENCE_ALIGNMENT_THRESHOLD:
        return cleaned[:max_chars]

    if len(best_sentence) <= max_chars:
        return best_sentence
    return best_sentence[:max_chars]
