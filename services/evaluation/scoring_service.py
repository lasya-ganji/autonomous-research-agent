from typing import List, Dict
from models.search_models import SearchResult
from services.evaluation.weight_service import get_dynamic_weights
from urllib.parse import urlparse
from datetime import datetime
import re
import math
import numpy as np
from collections import Counter
from services.retrieval.embedding_service import get_embedding


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


def compute_relevance(result: SearchResult, query: str, query_emb=None) -> float:
    title = (result.title or "")
    snippet = (result.snippet or "")
    content = (result.content or "")

    doc_text = f"{title} {snippet} {content}".strip().lower()
    query = query.strip().lower()

    if not doc_text or not query:
        return 0.0

    query_terms = query.split()
    doc_terms = doc_text.split()

    # KEYWORD SCORE (IMPROVED)
    doc_tf = Counter(doc_terms)

    keyword_score = 0.0
    for term in query_terms:
        if term in doc_tf:
            # controlled TF boost (avoid explosion)
            keyword_score += min(doc_tf[term] / 3, 1.0)

    keyword_score = keyword_score / (len(query_terms) or 1)

    # SEMANTIC SCORE
    if query_emb is None:
        query_emb = get_embedding(query)

    doc_emb = get_embedding(doc_text[:1000])

    if query_emb is not None and doc_emb is not None:
        q = np.array(query_emb)
        d = np.array(doc_emb)

        denom = np.linalg.norm(q) * np.linalg.norm(d)
        semantic_score = float(np.dot(q, d) / denom) if denom > 0 else 0.0

        # normalize safely
        semantic_score = max(0.0, min(semantic_score, 1.0))
    else:
        semantic_score = 0.0

    # TITLE MATCH SCORE
    title_words = set(title.lower().split())
    query_words = set(query_terms)

    title_score = len(query_words & title_words) / (len(query_words) or 1)

    # BASE COMBINATION
    base = (
        0.65 * semantic_score +
        0.25 * keyword_score +
        0.10 * title_score
    )

    # CONTROLLED BOOSTING
    if semantic_score > 0.6 and keyword_score > 0.5:
        base += 0.05

    if semantic_score > 0.6 and title_score > 0.4:
        base += 0.03

    if keyword_score > 0.7 and semantic_score < 0.3:
        base -= 0.05

    # FINAL NORMALIZATION
    base = max(0.0, min(base, 1.0))

    return round(base, 3)


HIGH_AUTHORITY = [
    "ieee.org", "acm.org", "nature.com", "sciencedirect.com",
    "springer.com", "mit.edu", "stanford.edu", "harvard.edu",
    "nasa.gov", "who.int", "oecd.org", "worldbank.org"
]

RESEARCH_PLATFORMS = [
    "arxiv.org", "researchgate.net", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov"
]

TRUSTED_KNOWLEDGE = [
    "wikipedia.org", "britannica.com"
]

TECH_BLOGS = [
    "medium.com", "towardsdatascience.com", "substack.com", "hashnode.dev"
]

NEWS_SOURCES = [
    "bbc.com", "nytimes.com", "reuters.com", "theguardian.com"
]

LOW_QUALITY = [
    "quora.com", "reddit.com", "pinterest.com"
]


def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc.lower()

        # remove www
        domain = domain.replace("www.", "")

        # HIGH AUTHORITY
        if any(d in domain for d in HIGH_AUTHORITY):
            return 0.95

        # RESEARCH
        if any(d in domain for d in RESEARCH_PLATFORMS):
            return 0.9

        # GOV / EDU fallback
        if domain.endswith(".gov") or domain.endswith(".edu"):
            return 0.93

        # TRUSTED KNOWLEDGE
        if any(d in domain for d in TRUSTED_KNOWLEDGE):
            return 0.85

        # NEWS
        if any(d in domain for d in NEWS_SOURCES):
            return 0.8

        # TECH BLOGS
        if any(d in domain for d in TECH_BLOGS):
            return 0.7

        # LOW QUALITY
        if any(d in domain for d in LOW_QUALITY):
            return 0.5

        # Heuristic fallback (IMPORTANT)
        if any(x in domain for x in ["blog", "dev", "tech"]):
            return 0.65

        return 0.75  

    except:
        return 0.5



def compute_recency(result: SearchResult, weight: float) -> float:
    today = datetime.now()
    days_old = None

    # 1. Extract publish date
    if getattr(result, "publish_date", None):
        try:
            pub_date = result.publish_date

            if isinstance(pub_date, str):
                pub_date = datetime.fromisoformat(pub_date[:10])

            # FUTURE DATE GUARD
            if pub_date > today:
                days_old = 0
            else:
                days_old = (today - pub_date).days

        except Exception:
            days_old = None

    # 2. Fallback: extract year
    if days_old is None:
        text = f"{result.title or ''} {result.snippet or ''}"
        years = re.findall(r"(20\d{2})", text)

        if years:
            latest_year = max(map(int, years))
            year_diff = today.year - latest_year

            # clamp year diff
            year_diff = max(0, min(year_diff, 20))  # max 20 years

            days_old = year_diff * 365

    # 3. Final fallback
    if days_old is None:
        return 0.5  # neutral score

    # 4. Clamp days_old 
    days_old = max(0, min(days_old, 3650))  # max ~10 years

    # 5. Adaptive decay window
    if weight >= 0.3:
        max_days = 365       # very time-sensitive
    elif weight >= 0.25:
        max_days = 730       # moderate
    elif weight >= 0.2:
        max_days = 1200      # semi-static
    else:
        max_days = 2000      # evergreen

    # 6. Compute recency
    recency = math.exp(-days_old / max_days)

    # 7. Smooth lower bound
    recency = max(0.1, recency) 

    # 8. Ensure valid range
    recency = min(max(recency, 0), 1)

    return round(recency, 3)


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


def is_outdated(result: SearchResult, recency_weight: float) -> bool:
    if recency_weight < 0.30:
        return False

    current_year = datetime.now().year

    if getattr(result, "publish_date", None):
        try:
            year = int(str(result.publish_date)[:4])
            return (current_year - year) > 3
        except:
            pass

    text = f"{result.title or ''} {result.snippet or ''}"
    years = re.findall(r"(20\d{2})", text)

    if years:
        latest = max(map(int, years))
        return (current_year - latest) > 3

    return False


def score_results(results: List[SearchResult], query: str) -> List[SearchResult]:
    weights = validate_weights(get_dynamic_weights(query))
    query_emb = get_embedding(query)

    scored = []

    for r in results:

        if is_outdated(r, weights["recency"]):
            continue

        r.relevance_score = compute_relevance(r, query, query_emb)
        r.domain_score = compute_domain(r)
        r.recency_score = compute_recency(r, weights["recency"])
        r.depth_score = compute_depth(r)

        # relevance-dominant formulation
        # compute secondary signals
        secondary = (
            weights["domain"] * r.domain_score +
            weights["recency"] * r.recency_score +
            weights["depth"] * r.depth_score
        )

        # balanced scoring 
        r.quality_score = (
            0.7 * r.relevance_score +
            0.3 * secondary
        )

        # clamp + stability
        r.quality_score = round(max(0.2, min(r.quality_score, 1.0)), 3)
        scored.append(r)

    scored.sort(key=lambda x: x.quality_score, reverse=True)

    scored = [r for r in scored if r.quality_score >= 0.25]

    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored