from typing import List, Dict
from models.search_models import SearchResult
from urllib.parse import urlparse


# DEFAULT WEIGHTS (fallback)

DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}


# VALIDATION LAYER (partial)

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:

    # Normalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Minimum thresholds (anchoring)
    weights["relevance"] = max(weights["relevance"], 0.35)
    weights["recency"] = max(weights["recency"], 0.15)
    weights["domain"] = max(weights["domain"], 0.15)
    weights["depth"] = max(weights["depth"], 0.05)

    # Renormalize again
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


# SIGNALS

def compute_relevance(result: SearchResult, query: str) -> float:
    query_words = set(query.lower().split())
    title_words = set(result.title.lower().split())
    snippet_words = set(result.snippet.lower().split())

    # overlap score
    title_overlap = len(query_words & title_words) / len(query_words)
    snippet_overlap = len(query_words & snippet_words) / len(query_words)

    score = 0.6 * title_overlap + 0.4 * snippet_overlap

    return min(score, 1.0)


def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc

        if "gov" in domain or "edu" in domain or "nih" in domain:
            return 0.9
        elif "org" in domain:
            return 0.8
        else:
            return 0.6

    except:
        return 0.5


def compute_recency(result: SearchResult) -> float:
    # No date yet → neutral (future upgrade)
    return 0.5


def compute_depth(result: SearchResult) -> float:
    length = len(result.snippet)

    if length > 300:
        return 0.9
    elif length > 150:
        return 0.7
    else:
        return 0.5


# MAIN SCORING FUNCTION
def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:

    weights = validate_weights(DEFAULT_WEIGHTS)

    for r in results:
        r.relevance_score = compute_relevance(r, query)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r)
        r.depth_score = compute_depth(r)

        # Final weighted score
        final_score = (
            weights["relevance"] * r.relevance_score +
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        r.quality_score = final_score

    # Sort
    results.sort(key=lambda x: x.quality_score, reverse=True)

    # Assign rank
    for i, r in enumerate(results):
        r.rank = i + 1

    return results