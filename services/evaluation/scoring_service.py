from typing import List, Dict
from models.search_models import SearchResult
from services.evaluation.weight_service import get_dynamic_weights
from urllib.parse import urlparse
from datetime import datetime
import re


# CONFIG 

SCORING_CONFIG = {
    "min_quality_threshold": 0.3,
    "recency_days": {
        "high": 730,     
        "medium": 1000,
        "low": 1500
    },
    "max_recency_year_gap": 3
}


# DEFAULT WEIGHTS (fallback)

DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}


# URL NORMALIZATION

def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except Exception:
        return url


# DEDUPLICATION (Stage 1 & 2)

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    seen_urls = set()
    unique_results = []

    for r in results:
        if not r.url:
            continue

        norm_url = normalize_url(r.url)

        if norm_url not in seen_urls:
            seen_urls.add(norm_url)
            r.url = norm_url
            unique_results.append(r)

    return unique_results


# VALIDATION LAYER

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    required = {"relevance", "recency", "domain", "depth"}

    if not weights or not all(k in weights for k in required):
        return DEFAULT_WEIGHTS

    # normalize
    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    # minimum thresholds (signal anchoring)
    weights["relevance"] = max(weights["relevance"], 0.35)
    weights["recency"] = max(weights["recency"], 0.15)
    weights["domain"] = max(weights["domain"], 0.15)
    weights["depth"] = max(weights["depth"], 0.05)

    # max cap (dominance control)
    weights = {k: min(v, 0.6) for k, v in weights.items()}

    # re-normalize
    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    return weights


# SIGNALS

def compute_relevance(result: SearchResult, query: str) -> float:
    query_words = set((query or "").lower().split())
    title_words = set((result.title or "").lower().split())
    snippet_words = set((result.snippet or "").lower().split())

    # keyword overlap
    keyword_score = len(query_words & snippet_words) / (len(query_words) or 1)

    # title match
    title_score = len(query_words & title_words) / (len(query_words) or 1)

    # placeholder for embeddings (future)
    semantic_score = getattr(result, "semantic_score", 0.0)

    relevance = (
        0.5 * semantic_score +
        0.3 * keyword_score +
        0.2 * title_score
    )

    return round(min(relevance, 1.0), 3)


def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc.lower()

        if any(x in domain for x in ["gov", "edu", "nih", "who", "ieee", "acm"]):
            return 0.95

        if any(x in domain for x in ["org", "docs", "developer", "openai", "aws", "google"]):
            return 0.85

        if any(x in domain for x in ["wikipedia", "britannica"]):
            return 0.8

        if any(x in domain for x in ["medium", "substack", "blog"]):
            return 0.65

        if any(x in domain for x in ["reddit", "quora", "stackexchange"]):
            return 0.55

        return 0.7

    except Exception:
        return 0.5


def compute_recency(result: SearchResult, recency_weight: float) -> float:
    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return 0.5

    latest_year = max(map(int, years))
    current_year = datetime.now().year

    days_old = (current_year - latest_year) * 365

    # dynamic max_days (PRD aligned)
    if recency_weight >= 0.30:
        max_days = SCORING_CONFIG["recency_days"]["high"]
    elif recency_weight >= 0.20:
        max_days = SCORING_CONFIG["recency_days"]["medium"]
    else:
        max_days = SCORING_CONFIG["recency_days"]["low"]

    recency = max(0, 1 - (days_old / max_days))
    return round(min(recency, 1.0), 3)


def compute_depth(result: SearchResult) -> float:
    text = result.snippet or ""

    if not text.strip():
        return 0.0

    words = text.split()
    word_count = len(words)

    # length score
    if word_count > 80:
        length_score = 0.9
    elif word_count > 50:
        length_score = 0.7
    elif word_count > 25:
        length_score = 0.5
    else:
        length_score = 0.3

    # sentence complexity
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_len = word_count / len(sentences) if sentences else word_count

    if avg_len > 15:
        complexity_score = 0.9
    elif avg_len > 10:
        complexity_score = 0.7
    elif avg_len > 6:
        complexity_score = 0.5
    else:
        complexity_score = 0.3

    # density
    density = len(set(words)) / word_count if word_count else 0

    if density > 0.7:
        density_score = 0.9
    elif density > 0.5:
        density_score = 0.7
    elif density > 0.3:
        density_score = 0.5
    else:
        density_score = 0.3

    depth = (
        0.4 * length_score +
        0.3 * complexity_score +
        0.3 * density_score
    )

    return round(min(depth, 1.0), 3)


# HARD CONSTRAINT (RECENCY)

def is_outdated(result: SearchResult, recency_weight: float) -> bool:
    if recency_weight < 0.3:
        return False

    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return False

    latest_year = max(map(int, years))
    current_year = datetime.now().year

    return (current_year - latest_year) > SCORING_CONFIG["max_recency_year_gap"]


# MAIN SCORING FUNCTION

def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:

    # Step 1: Deduplication
    results = deduplicate_results(results)

    # Step 2: Dynamic weights
    weights = validate_weights(get_dynamic_weights(query))

    scored_results = []

    for r in results:

        # Step 3: Hard constraint
        if is_outdated(r, weights["recency"]):
            continue

        # Step 4: Compute signals
        r.relevance_score = compute_relevance(r, query)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r, weights["recency"])
        r.depth_score = compute_depth(r)

        # Step 5: Final score
        r.quality_score = (
            weights["relevance"] * r.relevance_score +
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        scored_results.append(r)

    # Step 6: Sort
    scored_results.sort(key=lambda x: x.quality_score, reverse=True)

    # Step 7: Filter low-quality
    threshold = SCORING_CONFIG["min_quality_threshold"]
    scored_results = [r for r in scored_results if r.quality_score > threshold]

    # Step 8: Ranking
    for i, r in enumerate(scored_results):
        r.rank = i + 1

    return scored_results