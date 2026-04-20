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

from config.constants.scoring_constants import (
    DEFAULT_WEIGHTS,
    MIN_THRESHOLDS,
    SEMANTIC_WEIGHT,
    KEYWORD_WEIGHT,
    TITLE_WEIGHT,
    SEMANTIC_BOOST_THRESHOLD,
    KEYWORD_BOOST_THRESHOLD,
    TITLE_BOOST_THRESHOLD,
    MAX_TF_NORMALIZATION,
    RELEVANCE_MIX,
    SECONDARY_MIX,
    QUALITY_SCORE_MIN,
    QUALITY_SCORE_FILTER,
    DEPTH_HIGH_WORDS, DEPTH_MED_WORDS, DEPTH_LOW_WORDS,
    DEPTH_HIGH_SCORE, DEPTH_MED_SCORE, DEPTH_LOW_SCORE, DEPTH_MIN_SCORE,
    RECENCY_HIGH_WEIGHT, RECENCY_MED_WEIGHT, RECENCY_LOW_WEIGHT,
    RECENCY_HIGH_DAYS, RECENCY_MED_DAYS, RECENCY_LOW_DAYS, RECENCY_STATIC_DAYS,
    RECENCY_MIN_SCORE, OUTDATED_YEAR_THRESHOLD,
    RELEVANCE_DUAL_BOOST, RELEVANCE_TITLE_BOOST, RELEVANCE_KEYWORD_PENALTY,
    WEIGHT_MAX_CAP,
    DOC_EMBED_CHAR_LIMIT,
    RECENCY_MAX_YEAR_DIFF,
    RECENCY_DAYS_PER_YEAR,
    RECENCY_MAX_DAYS_OLD,
    DOMAIN_SCORE_GOV_EDU, DOMAIN_SCORE_BLOG_SUBDOMAIN, DOMAIN_SCORE_DEFAULT,
    DOMAIN_BLOG_SUBSTRINGS,
    DOMAIN_TIERS, ACADEMIC_TLDS, GOV_TLDS,
)

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return DEFAULT_WEIGHTS

    total = sum(weights.values()) or 1
    weights = {k: v / total for k, v in weights.items()}

    for k in weights:
        weights[k] = max(MIN_THRESHOLDS.get(k, 0), min(weights[k], WEIGHT_MAX_CAP))

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

    # -------------------------------
    # KEYWORD SCORE 
    # -------------------------------
    doc_tf = Counter(doc_terms)

    keyword_score = 0.0
    for term in query_terms:
        if term in doc_tf:
            keyword_score += min(doc_tf[term] / MAX_TF_NORMALIZATION, 1.0)

    keyword_score = keyword_score / max(len(query_terms), 1)

    # -------------------------------
    # SEMANTIC SCORE
    # -------------------------------
    if query_emb is None:
        query_emb = get_embedding(query)

    doc_emb = get_embedding(doc_text[:DOC_EMBED_CHAR_LIMIT])

    if query_emb is not None and doc_emb is not None:
        q = np.array(query_emb)
        d = np.array(doc_emb)

        denom = np.linalg.norm(q) * np.linalg.norm(d)
        semantic_score = float(np.dot(q, d) / denom) if denom > 0 else 0.0

        semantic_score = max(0.0, min(semantic_score, 1.0))
    else:
        semantic_score = 0.0

    # -------------------------------
    # TITLE SCORE
    # -------------------------------
    title_words = set(title.lower().split())
    query_words = set(query_terms)

    title_score = len(query_words & title_words) / (len(query_words) or 1)

    base = (
        SEMANTIC_WEIGHT * semantic_score +
        KEYWORD_WEIGHT * keyword_score +
        TITLE_WEIGHT * title_score
    )

    if semantic_score > SEMANTIC_BOOST_THRESHOLD and keyword_score > KEYWORD_BOOST_THRESHOLD:
        base += RELEVANCE_DUAL_BOOST

    if semantic_score > SEMANTIC_BOOST_THRESHOLD and title_score > TITLE_BOOST_THRESHOLD:
        base += RELEVANCE_TITLE_BOOST

    if keyword_score > KEYWORD_BOOST_THRESHOLD and semantic_score < SEMANTIC_WEIGHT:
        base -= RELEVANCE_KEYWORD_PENALTY

    return round(max(0.0, min(base, 1.0)), 3)


def compute_domain(result: SearchResult) -> float:
    try:
        domain = urlparse(str(result.url)).netloc.lower().replace("www.", "")

        # Curated tiers from domain_authority.json — highest authority first.
        # To add a new site: edit config/domain_authority.json, no code change needed.
        for tier in DOMAIN_TIERS:
            if any(d in domain for d in tier["domains"]):
                return tier["score"]

        # Structural TLD signals — catch academic/gov institutions not in any curated list.
        # Examples: cam.ac.uk (Cambridge), ox.ac.uk (Oxford), csiro.edu.au, nih.gov, gc.ca
        if any(domain.endswith(tld) for tld in ACADEMIC_TLDS):
            return DOMAIN_SCORE_GOV_EDU
        if any(domain.endswith(tld) for tld in GOV_TLDS):
            return DOMAIN_SCORE_GOV_EDU

        # Subdomain heuristic: blog.*, dev.*, tech.* prefixes indicate informal content
        if any(x in domain for x in DOMAIN_BLOG_SUBSTRINGS):
            return DOMAIN_SCORE_BLOG_SUBDOMAIN

        return DOMAIN_SCORE_DEFAULT

    except Exception:
        return DOMAIN_SCORE_DEFAULT


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
            year_diff = max(0, min(year_diff, RECENCY_MAX_YEAR_DIFF))  # max 20 years

            days_old = year_diff * RECENCY_DAYS_PER_YEAR

    # 3. Final fallback
    if days_old is None:
        return 0.5  # neutral score

    # 4. Clamp days_old 
    days_old = max(0, min(days_old, RECENCY_MAX_DAYS_OLD))  # max ~10 years

    # 5. Adaptive decay window
    if weight >= RECENCY_HIGH_WEIGHT:
        max_days = RECENCY_HIGH_DAYS
    elif weight >= RECENCY_MED_WEIGHT:
        max_days = RECENCY_MED_DAYS
    elif weight >= RECENCY_LOW_WEIGHT:
        max_days = RECENCY_LOW_DAYS
    else:
        max_days = RECENCY_STATIC_DAYS

    # 6. Compute recency
    recency = math.exp(-days_old / max_days)

    # 7. Smooth lower bound
    recency = max(RECENCY_MIN_SCORE, recency) 

    # 8. Ensure valid range
    recency = min(max(recency, 0), 1)

    return round(recency, 3)


def compute_depth(result: SearchResult) -> float:
    text = result.content or result.snippet or ""
    wc = len(text.split())

    if wc > DEPTH_HIGH_WORDS:
        return DEPTH_HIGH_SCORE
    elif wc > DEPTH_MED_WORDS:
        return DEPTH_MED_SCORE
    elif wc > DEPTH_LOW_WORDS:
        return DEPTH_LOW_SCORE
    else:
        return DEPTH_MIN_SCORE


def is_outdated(result: SearchResult, recency_weight: float) -> bool:
    if recency_weight < RECENCY_HIGH_WEIGHT:
        return False

    current_year = datetime.now().year

    if getattr(result, "publish_date", None):
        try:
            year = int(str(result.publish_date)[:4])
            return (current_year - year) > OUTDATED_YEAR_THRESHOLD
        except:
            pass

    text = f"{result.title or ''} {result.snippet or ''}"
    years = re.findall(r"(20\d{2})", text)

    if years:
        latest = max(map(int, years))
        return (current_year - latest) > OUTDATED_YEAR_THRESHOLD

    return False


def score_results(results: List[SearchResult], query: str, state=None) -> List[SearchResult]:
    weights = validate_weights(get_dynamic_weights(query,state))
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
            RELEVANCE_MIX * r.relevance_score +
            SECONDARY_MIX * secondary
        )

        # clamp + stability
        r.quality_score = round(max(QUALITY_SCORE_MIN, min(r.quality_score, 1.0)), 3)
        scored.append(r)
        
        print(f"[SCORING] URL={r.url} | rel={r.relevance_score:.3f} dom={r.domain_score:.3f} rec={r.recency_score:.3f} dep={r.depth_score:.3f} => quality={r.quality_score:.3f}")

    scored.sort(key=lambda x: x.quality_score, reverse=True)

    scored = [r for r in scored if r.quality_score >= QUALITY_SCORE_FILTER]

    for i, r in enumerate(scored):
        r.rank = i + 1

    return scored