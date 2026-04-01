from typing import List, Dict
from models.search_models import SearchResult
from services.evaluation.weight_service import get_dynamic_weights
from urllib.parse import urlparse
import re
from datetime import datetime

from services.retrieval.embedding_service import get_embedding
from sklearn.metrics.pairwise import cosine_similarity


# CONFIG
SCORING_CONFIG = {
    "min_quality_threshold": 0.35,
    "max_recency_year_gap": 3
}


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


# DEDUPLICATION
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


# VALIDATION
def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    required = {"relevance", "recency", "domain", "depth"}

    if not weights or not all(k in weights for k in required):
        return DEFAULT_WEIGHTS

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    weights["relevance"] = max(weights["relevance"], 0.35)
    weights["recency"] = max(weights["recency"], 0.15)
    weights["domain"] = max(weights["domain"], 0.15)
    weights["depth"] = max(weights["depth"], 0.05)

    weights = {k: min(v, 0.6) for k, v in weights.items()}

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    return weights


# HYBRID RELEVANCE
def compute_relevance(result: SearchResult, query: str, query_emb=None) -> float:

    doc_text = (
        result.content
        or result.snippet
        or result.title
        or ""
    )

    semantic = 0.0

    if query_emb is not None:
        try:
            doc_emb = get_embedding(doc_text[:1000])
            sim = cosine_similarity([query_emb], [doc_emb])[0][0]
            semantic = (sim + 1) / 2
        except Exception:
            semantic = 0.0

    query_words = set(query.lower().split())
    doc_words = set(doc_text.lower().split())

    keyword = len(query_words & doc_words) / (len(query_words) or 1)

    title_words = set((result.title or "").lower().split())
    title_score = len(query_words & title_words) / (len(query_words) or 1)

    base = (
        0.6 * semantic +
        0.25 * keyword +
        0.15 * title_score
    )

    adjustment = 0.0

    if semantic > 0.6 and keyword > 0.6:
        adjustment += 0.05
    elif keyword > 0.6 and semantic < 0.3:
        adjustment -= 0.05

    if semantic > 0.6 and title_score > 0.6:
        adjustment += 0.05

    return max(0.0, min(base + adjustment, 1.0))

# DOMAIN
def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc.lower()

        if any(x in domain for x in ["gov", "edu", "ieee", "acm"]):
            return 0.95
        if any(x in domain for x in ["org", "docs", "developer", "openai", "aws", "google"]):
            return 0.85
        if "wikipedia" in domain:
            return 0.8
        if any(x in domain for x in ["medium", "blog"]):
            return 0.65
        if any(x in domain for x in ["reddit", "quora"]):
            return 0.55

        return 0.7

    except:
        return 0.5

# RECENCY
def compute_recency(result: SearchResult) -> float:
    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return 0.5

    latest = max(map(int, years))
    current_year = datetime.now().year

    gap = current_year - latest

    if gap <= 1:
        return 0.9
    elif gap <= 3:
        return 0.75
    elif gap <= 5:
        return 0.6
    else:
        return 0.4


#  OUTDATED FILTER 

def is_outdated(result: SearchResult, recency_weight: float) -> bool:
    if recency_weight < 0.25:
        return False

    text = (result.snippet or "") + " " + (result.title or "")
    years = re.findall(r"(20\d{2})", text)

    if not years:
        return False

    latest = max(map(int, years))
    current_year = datetime.now().year

    return (current_year - latest) > SCORING_CONFIG["max_recency_year_gap"]


# DEPTH
def compute_depth(result: SearchResult) -> float:
    text = result.snippet or ""
    length = len(text)

    if length > 600:
        base = 0.9
    elif length > 200:
        base = 0.7
    elif length > 100:
        base = 0.55
    else:
        base = 0.35

    if re.search(r"\d+", text):
        base += 0.05

    if any(x in text for x in ["-", "•", ":", ";"]):
        base += 0.05

    return min(base, 1.0)

# MAIN
def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:

    results = deduplicate_results(results)

    weights = validate_weights(get_dynamic_weights(query))

    scored = []

    try:
        query_emb = get_embedding(query)
    except:
        query_emb = None

    for r in results:

        # skip outdated
        if is_outdated(r, weights["recency"]):
            continue

        r.relevance_score = compute_relevance(r, query, query_emb)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r)
        r.depth_score = compute_depth(r)

        base_score = (
            weights["relevance"] * r.relevance_score +
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        #BOOST GOOD RESULTS
        r.quality_score = base_score

        scored.append(r)

    scored.sort(key=lambda x: x.quality_score, reverse=True)
    print(f"[SCORING DEBUG] Before filter: {len(results)}, After scoring: {len(scored)}")

    # stronger filtering 
    scored = [r for r in scored if r.quality_score > SCORING_CONFIG["min_quality_threshold"]]

    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored