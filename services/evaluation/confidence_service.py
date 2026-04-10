from typing import List
from models.search_models import SearchResult
import numpy as np
from services.retrieval.embedding_service import get_embedding


TOP_K = 5


def compute_confidence(results: List[SearchResult], query: str) -> float:
    """
    Improved confidence model:

    - quality → are sources good?
    - coverage → do sources answer query?
    - agreement → are sources coherent?
    - diversity → are sources non-redundant?
    """

    if not results:
        return 0.0

    # top-k selection
    results = sorted(results, key=lambda x: x.quality_score, reverse=True)[:TOP_K]
    scores = [r.quality_score for r in results]

    # QUALITY
    quality = sum(scores) / len(scores)

    # EMBEDDINGS
    query_emb = get_embedding(query)

    doc_embeddings = []
    for r in results:
        text = (r.content or r.snippet or r.title or "")[:1000]
        emb = get_embedding(text)
        if emb:
            doc_embeddings.append(np.array(emb))

    if not doc_embeddings or query_emb is None:
        return round(quality, 3)

    query_vec = np.array(query_emb)

    # COVERAGE
    # how well docs align with query
    coverage_scores = []
    for emb in doc_embeddings:
        denom = np.linalg.norm(query_vec) * np.linalg.norm(emb)
        if denom > 0:
            sim = float(np.dot(query_vec, emb) / denom)
            coverage_scores.append(max(0.0, min(sim, 1.0)))

    coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0

    # AGREEMENT (CENTROID BASED)
    centroid = np.mean(doc_embeddings, axis=0)

    agreement_scores = []
    for emb in doc_embeddings:
        denom = np.linalg.norm(centroid) * np.linalg.norm(emb)
        if denom > 0:
            sim = float(np.dot(centroid, emb) / denom)
            agreement_scores.append(max(0.0, min(sim, 1.0)))

    agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0

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

    redundancy = sum(pairwise_sims) / len(pairwise_sims) if pairwise_sims else 0.0
    diversity = 1 - redundancy

    # FINAL CONFIDENCE
    confidence = (
        0.5 * quality +
        0.2 * coverage +
        0.2 * agreement +
        0.1 * diversity
    )

    # debug
    print(f"[CONFIDENCE] quality={quality:.3f} coverage={coverage:.3f} agreement={agreement:.3f} diversity={diversity:.3f} => final={confidence:.3f}")

    return round(max(0.0, min(confidence, 1.0)), 3)