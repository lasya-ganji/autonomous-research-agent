from typing import List
from models.search_models import SearchResult


def compute_confidence(results: List[SearchResult]) -> float:
    """
    Computes confidence score based on PRD:
    - average_score
    - score consistency (variance)
    - top-k margin
    """

    if not results:
        return 0.0

    scores = [r.quality_score for r in results]

    if not scores:
        return 0.0

    # average score
    avg_score = sum(scores) / len(scores)

    # variance (consistency)
    mean = avg_score
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    score_consistency = 1 - variance  # normalized

    # top-k margin
    sorted_scores = sorted(scores, reverse=True)
    if len(sorted_scores) > 1:
        top_k_margin = sorted_scores[0] - sorted_scores[1]
    else:
        top_k_margin = sorted_scores[0]

    # simple approximation (no embeddings yet)
    confidence = (
        0.5 * avg_score +
        0.3 * score_consistency +
        0.2 * top_k_margin
    )

    return max(0.0, min(confidence, 1.0))