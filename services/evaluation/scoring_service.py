from typing import List, Dict
from models.search_models import SearchResult
from services.evaluation.weight_service import get_dynamic_weights
from urllib.parse import urlparse
from datetime import datetime
import re


DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}


MIN_THRESHOLDS = {
    "relevance": 0.35,
    "recency": 0.1,
    "domain": 0.1,
    "depth": 0.05
}


# WEIGHT VALIDATION (SOFT)

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:

    if not weights:
        return DEFAULT_WEIGHTS

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    for k in weights:
        weights[k] = max(MIN_THRESHOLDS.get(k, 0), min(weights[k], 0.6))

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    return weights


# RELEVANCE (STRONG FALLBACK)

def compute_relevance(result: SearchResult, query: str) -> float:

    query_words = set(query.lower().split())

    title = (result.title or "").lower()
    snippet = (result.snippet or "").lower()
    content = (result.content or "").lower()

    combined = f"{title} {snippet} {content}"
    words = set(combined.split())

    # keyword overlap
    keyword = len(query_words & words) / (len(query_words) or 1)

    # title match
    title_words = set(title.split())
    title_score = len(query_words & title_words) / (len(query_words) or 1)

    # pseudo-semantic (important before embeddings)
    coverage = min(len(words) / 200, 1.0)

    base = 0.6 * keyword + 0.2 * title_score + 0.2 * coverage

    # adjustment
    if keyword > 0.6 and title_score > 0.4:
        base += 0.05

    return round(min(base, 1.0), 3)


# DOMAIN

def compute_domain(result: SearchResult) -> float:

    try:
        domain = urlparse(str(result.url)).netloc.lower()

        if any(x in domain for x in ["gov", "edu", "oecd", "acm", "ieee"]):
            return 0.95

        if any(x in domain for x in ["google", "openai", "mit", "harvard"]):
            return 0.9

        if "wikipedia" in domain:
            return 0.8

        if any(x in domain for x in ["medium", "blog"]):
            return 0.65

        return 0.7

    except:
        return 0.5


# RECENCY

def compute_recency(result: SearchResult, weight: float) -> float:

    current_year = datetime.now().year

    # 1. PRIMARY: USE METADATA
    if getattr(result, "publish_date", None):
        try:
            year = int(str(result.publish_date)[:4])
            gap = current_year - year
        except:
            gap = None
    else:
        gap = None

    # 2. FALLBACK: REGEX (TEXT)
    if gap is None:
        text = f"{result.title or ''} {result.snippet or ''}"
        years = re.findall(r"(20\d{2})", text)

        if years:
            latest = max(map(int, years))
            gap = current_year - latest

    # 3. DEFAULT (NO INFO)
    if gap is None:
        return 0.5

    # 4. SCORING
    if gap <= 1:
        return 1.0
    elif gap <= 2:
        return 0.8
    elif gap <= 3:
        return 0.6
    else:
        return 0.3


# DEPTH

def compute_depth(result: SearchResult) -> float:

    text = result.content or result.snippet or ""

    wc = len(text.split())

    if wc > 1200:
        return 1.0
    elif wc > 700:
        return 0.75
    elif wc > 300:
        return 0.55
    else:
        return 0.3


# HARD FILTER

def is_outdated(result: SearchResult, recency_weight: float) -> bool:

    if recency_weight < 0.30:
        return False

    current_year = datetime.now().year

    # 1. Try metadata
    if getattr(result, "publish_date", None):
        try:
            year = int(str(result.publish_date)[:4])
            return (current_year - year) > 3
        except:
            pass

    # 2. Fallback regex
    text = f"{result.title or ''} {result.snippet or ''}"
    years = re.findall(r"(20\d{2})", text)

    if years:
        latest = max(map(int, years))
        return (current_year - latest) > 3

    return False

# MAIN

def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:

    weights = validate_weights(get_dynamic_weights(query))

    scored = []

    for r in results:

        if is_outdated(r, weights["recency"]):
            continue

        r.relevance_score = compute_relevance(r, query)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r, weights["recency"])
        r.depth_score = compute_depth(r)

        r.quality_score = (
            weights["relevance"] * r.relevance_score +
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        scored.append(r)

    # sort
    scored.sort(key=lambda x: x.quality_score, reverse=True)

    # soft filter
    scored = [r for r in scored if r.quality_score >= 0.25]

    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored