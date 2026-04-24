from typing import List
import numpy as np

from models.search_models import SearchResult
from services.retrieval.embedding_service import get_embedding

from config.constants.confidence_constants import (
    TOP_K,
    QUALITY_WEIGHT,
    COVERAGE_WEIGHT,
    AGREEMENT_WEIGHT,
    DIVERSITY_WEIGHT,
    MIN_CONFIDENCE,
    MAX_CONFIDENCE,
    MAX_TEXT_LENGTH
)


def compute_confidence(results: List[SearchResult], query: str) -> float:
    """
    Confidence Model 

    Signals:
    - quality   → source reliability
    - coverage  → query alignment
    - agreement → consistency across sources
    - diversity → non-redundancy

    Returns:
    - confidence score in [0, 1]
    """

    if not results:
        return 0.0

    # TOP-K SELECTION
    results = sorted(
        results,
        key=lambda x: getattr(x, "quality_score", 0.0),
        reverse=True
    )[:TOP_K]

    scores = [getattr(r, "quality_score", 0.0) for r in results]

    # QUALITY
    quality = sum(scores) / len(scores) if scores else 0.0

    # EMBEDDINGS
    query_emb = get_embedding(query)

    doc_embeddings = []
    for r in results:
        text = (r.content or r.snippet or r.title or "")[:MAX_TEXT_LENGTH]

        emb = get_embedding(text)
        if emb:
            doc_embeddings.append(np.array(emb))

    # fallback if embeddings unavailable
    if not doc_embeddings or query_emb is None:
        return round(quality, 3)

    query_vec = np.array(query_emb)

    # COVERAGE
    coverage_scores = []
    for emb in doc_embeddings:
        denom = np.linalg.norm(query_vec) * np.linalg.norm(emb)

        if denom > 0:
            sim = float(np.dot(query_vec, emb) / denom)
            sim = max(0.0, min(sim, 1.0))
            coverage_scores.append(sim)

    coverage = (
        sum(coverage_scores) / len(coverage_scores)
        if coverage_scores else 0.0
    )

    # AGREEMENT (CENTROID)
    centroid = np.mean(doc_embeddings, axis=0)

    agreement_scores = []
    for emb in doc_embeddings:
        denom = np.linalg.norm(centroid) * np.linalg.norm(emb)

        if denom > 0:
            sim = float(np.dot(centroid, emb) / denom)
            sim = max(0.0, min(sim, 1.0))
            agreement_scores.append(sim)

    agreement = (
        sum(agreement_scores) / len(agreement_scores)
        if agreement_scores else 0.0
    )

    # DIVERSITY
    pairwise_sims = []

    for i in range(len(doc_embeddings)):
        for j in range(i + 1, len(doc_embeddings)):
            a = doc_embeddings[i]
            b = doc_embeddings[j]

            denom = np.linalg.norm(a) * np.linalg.norm(b)

            if denom > 0:
                sim = float(np.dot(a, b) / denom)
                pairwise_sims.append(sim)

    redundancy = (
        sum(pairwise_sims) / len(pairwise_sims)
        if pairwise_sims else 0.0
    )

    diversity = 1 - redundancy

    # FINAL CONFIDENCE
    confidence = (
        QUALITY_WEIGHT * quality +
        COVERAGE_WEIGHT * coverage +
        AGREEMENT_WEIGHT * agreement +
        DIVERSITY_WEIGHT * diversity
    )

    # clamp
    confidence = max(MIN_CONFIDENCE, min(confidence, MAX_CONFIDENCE))

    # DEBUG LOG
    print(
        f"[CONFIDENCE] "
        f"quality={quality:.3f} "
        f"coverage={coverage:.3f} "
        f"agreement={agreement:.3f} "
        f"diversity={diversity:.3f} "
        f"=> final={confidence:.3f}"
    )

    return round(confidence, 3)