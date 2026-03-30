from typing import List, Dict
from models.search_models import SearchResult
from services.evaluation.weight_service import get_dynamic_weights
from urllib.parse import urlparse
import re


# DEFAULT WEIGHTS (fallback)
DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}


# DEDUPLICATION

def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    seen_urls = set()
    unique_results = []

    for r in results:
        if r.url and r.url not in seen_urls:
            seen_urls.add(r.url)
            unique_results.append(r)

    return unique_results


# VALIDATION LAYER

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    required = {"relevance", "recency", "domain", "depth"}

    if not weights or not all(k in weights for k in required):
        return DEFAULT_WEIGHTS

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    # minimum thresholds
    weights["relevance"] = max(weights["relevance"], 0.35)
    weights["recency"] = max(weights["recency"], 0.15)
    weights["domain"] = max(weights["domain"], 0.15)
    weights["depth"] = max(weights["depth"], 0.05)

    # max cap
    weights = {k: min(v, 0.6) for k, v in weights.items()}

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    return weights


# SIGNALS

def compute_relevance(result: SearchResult, query: str) -> float:
    query_words = set((query or "").lower().split())
    title_words = set((result.title or "").lower().split())
    snippet_words = set((result.snippet or "").lower().split())

    denom = len(query_words) or 1

    title_overlap = len(query_words & title_words) / denom
    snippet_overlap = len(query_words & snippet_words) / denom

    base = 0.6 * title_overlap + 0.4 * snippet_overlap

    # adjustment for agreement/conflict
    if title_overlap > 0.6 and snippet_overlap > 0.6:
        base += 0.05
    elif title_overlap > 0.6 and snippet_overlap < 0.2:
        base -= 0.05

    return max(0.0, min(base, 1.0))


# improved domain scoring with tiered logic
def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc.lower()

        # high authority (research / official)
        if any(x in domain for x in ["gov", "edu", "nih", "who", "ieee", "acm"]):
            return 0.95

        # trusted orgs / documentation
        if any(x in domain for x in ["org", "docs", "developer", "openai", "aws", "google"]):
            return 0.85

        # well-known knowledge platforms
        if any(x in domain for x in ["wikipedia", "britannica"]):
            return 0.8

        # medium authority (blogs / tech sites)
        if any(x in domain for x in ["medium", "substack", "blog"]):
            return 0.65

        # low authority (forums / user-generated)
        if any(x in domain for x in ["reddit", "quora", "stackexchange"]):
            return 0.55

        # fallback
        return 0.7

    except Exception:
        return 0.5


def compute_recency(result: SearchResult) -> float:
    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return 0.5

    latest_year = max(map(int, years))

    if latest_year >= 2024:
        return 0.9
    elif latest_year >= 2022:
        return 0.7
    else:
        return 0.5


# improved depth scoring using content signals
def compute_depth(result: SearchResult) -> float:
    text = result.snippet or ""
    length = len(text)

    # base on length
    if length > 400:
        base = 0.9
    elif length > 200:
        base = 0.7
    elif length > 100:
        base = 0.5
    else:
        base = 0.3

    # boost if numbers/data present
    if re.search(r"\d+", text):
        base += 0.05

    # boost if structured (lists / punctuation)
    if any(x in text for x in ["-", "•", ":", ";"]):
        base += 0.05

    return min(base, 1.0)


# recency hard constraint
def is_outdated(result: SearchResult, recency_weight: float) -> bool:
    if recency_weight < 0.3:
        return False

    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return False

    latest = max(map(int, years))
    return latest < 2022


# MAIN SCORING FUNCTION

def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:

    # deduplicate
    results = deduplicate_results(results)

    # get weights
    weights = get_dynamic_weights(query)
    weights = validate_weights(weights)

    scored = []

    for r in results:

        # hard recency filter
        if is_outdated(r, weights["recency"]):
            continue

        # compute signals
        r.relevance_score = compute_relevance(r, query)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r)
        r.depth_score = compute_depth(r)

        # final score
        r.quality_score = (
            weights["relevance"] * r.relevance_score +
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        scored.append(r)

    # sort by quality
    scored.sort(key=lambda x: x.quality_score, reverse=True)

    # remove very low quality
    scored = [r for r in scored if r.quality_score > 0.3]

    # assign rank
    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored