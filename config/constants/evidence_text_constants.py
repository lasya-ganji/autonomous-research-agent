"""
Tunable values for evidence snippet cleaning and excerpt selection.

Used by utils/evidence_text.py — adjust thresholds and patterns here.
"""

from typing import FrozenSet

# --- excerpt length ---
DEFAULT_EVIDENCE_EXCERPT_MAX_CHARS: int = 900

# --- sentence selection inside a chunk (claim vs sentence overlap) ---
# If best sentence overlap falls below this, fall back to a prefix slice of cleaned text.
WEAK_SENTENCE_ALIGNMENT_THRESHOLD: float = 0.05

# --- tokenization for overlap scoring (_overlap_score) ---
# Tokens with length strictly less than this are excluded from the overlap set.
MIN_TOKEN_LENGTH: int = 3

EVIDENCE_STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "by", "with",
    "is", "are", "was", "were", "be", "as", "at", "that", "this", "it", "from",
})

# --- cleaning / splitting regex pattern strings (compiled in utils/evidence_text) ---
URL_PATTERN: str = r"https?://\S+"
MARKDOWN_IMAGE_PATTERN: str = r"!\[[^\]]*\]\([^)]+\)"
MARKDOWN_LINK_PATTERN: str = r"\[([^\]]+)\]\([^)]+\)"
BASE64_PATTERN: str = r"[A-Za-z0-9+/]{80,}={0,2}"
NON_WORD_HEAVY_PATTERN: str = r"[^\w\s.,;:!?()'-]"
WHITESPACE_PATTERN: str = r"\s+"
SENTENCE_SPLIT_PATTERN: str = r"(?<=[.!?])\s+"

# Bare image filenames that survive markdown cleaning (e.g. "960x0-651.jpg)")
IMAGE_FILENAME_PATTERN: str = r"\b\w[\w\-]*\.(?:jpg|jpeg|png|gif|webp|svg|ico|bmp|tiff?)[\w\-]*\b\)?"

# Repeated dash/pipe sequences from scraped HTML tables (e.g. "--- --- ---", "| H100 |")
TABLE_SEPARATOR_PATTERN: str = r"(?:-{2,}\s*){2,}|(?:\|\s*)+"

# Minimum word count for a sentence to be considered non-fragment
MIN_SENTENCE_WORDS: int = 4
