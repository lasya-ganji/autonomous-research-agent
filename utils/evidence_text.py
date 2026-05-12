import re

from config.constants.evidence_text_constants import (
    BASE64_PATTERN,
    DEFAULT_EVIDENCE_EXCERPT_MAX_CHARS,
    EVIDENCE_STOPWORDS,
    IMAGE_FILENAME_PATTERN,
    MARKDOWN_IMAGE_PATTERN,
    MARKDOWN_LINK_PATTERN,
    MIN_SENTENCE_WORDS,
    MIN_TOKEN_LENGTH,
    NON_WORD_HEAVY_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    TABLE_SEPARATOR_PATTERN,
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
_IMAGE_FILENAME_RE = re.compile(IMAGE_FILENAME_PATTERN, re.IGNORECASE)
_TABLE_SEPARATOR_RE = re.compile(TABLE_SEPARATOR_PATTERN)


def clean_evidence_text(text: str) -> str:
    if not text:
        return ""

    cleaned = _MARKDOWN_IMAGE_RE.sub(" ", text)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _BASE64_RE.sub(" ", cleaned)
    cleaned = _IMAGE_FILENAME_RE.sub(" ", cleaned)
    cleaned = _TABLE_SEPARATOR_RE.sub(" ", cleaned)
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

    raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]

    # Tier-1: word count + no leading digit token (image caption / table artefact)
    sentences = [
        s for s in raw_sentences
        if len(s.split()) >= MIN_SENTENCE_WORDS and not s.split()[0][0].isdigit()
    ]
    # Tier-2: word count only (keeps content even if it starts with a digit)
    if not sentences:
        sentences = [s for s in raw_sentences if len(s.split()) >= MIN_SENTENCE_WORDS]
    # Tier-3: nothing at all — pass everything through
    if not sentences:
        sentences = raw_sentences

    if not sentences:
        return _safe_truncate(_strip_leading_digits(cleaned), max_chars)

    if not claim_text.strip():
        best = max(sentences, key=len)
        return _safe_truncate(_strip_leading_digits(best), max_chars)

    scored = [(s, _overlap_score(claim_text, s)) for s in sentences]
    best_sentence, best_score = max(scored, key=lambda x: x[1])

    if best_score < WEAK_SENTENCE_ALIGNMENT_THRESHOLD:
        return _safe_truncate(_strip_leading_digits(cleaned), max_chars)

    return _safe_truncate(_strip_leading_digits(_expand_excerpt(best_sentence, sentences, max_chars)), max_chars)


def _safe_truncate(text: str, max_chars: int) -> str:
    """Truncate at word boundary to avoid mid-word cuts."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def _strip_leading_digits(text: str) -> str:
    """Remove standalone leading digit token left by scraped caption/list numbers."""
    return re.sub(r'^\d+\s+', '', text).strip()


def _expand_excerpt(best_sentence: str, sentences: list[str], max_chars: int) -> str:
    """
    Expand outward from the best sentence to fill the available character budget.

    Tries forward sentences first (most natural reading order), then backward
    (prior context), alternating until the budget is exhausted or sentences run out.
    """
    if best_sentence not in sentences:
        return best_sentence

    idx = sentences.index(best_sentence)
    result = best_sentence
    fwd = idx + 1
    bwd = idx - 1

    while True:
        added = False
        if fwd < len(sentences):
            candidate = result + " " + sentences[fwd]
            if len(candidate) <= max_chars:
                result = candidate
                fwd += 1
                added = True
        if bwd >= 0:
            candidate = sentences[bwd] + " " + result
            if len(candidate) <= max_chars:
                result = candidate
                bwd -= 1
                added = True
        if not added:
            break

    return result
