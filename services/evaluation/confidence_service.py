from typing import List
from models.search_models import SearchResult


def compute_confidence(results: List[SearchResult]) -> float:
    """
    Confidence based on:
    - average_score
    - agreement_between_sources (approx)
    - score_consistency
    - top_k_margin
    """

    if not results:
        return 0.0

    # Use only top-k
    results = sorted(results, key=lambda x: x.quality_score, reverse=True)[:5]

    scores = [r.quality_score for r in results]

    # 1. average score
    avg_score = sum(scores) / len(scores)

    # 2. consistency (variance)
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    score_consistency = max(0, 1 - variance)

    # 3. top-k margin
    if len(scores) > 1:
        top_k_margin = scores[0] - scores[1]
    else:
        top_k_margin = scores[0]

    # 4. agreement_between_sources (simple text overlap)
    def text_overlap(a: str, b: str) -> float:
        a_words = set((a or "").lower().split())
        b_words = set((b or "").lower().split())

        if not a_words or not b_words:
            return 0.0

        return len(a_words & b_words) / len(a_words | b_words)

    agreements = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            agreements.append(
                text_overlap(results[i].snippet, results[j].snippet)
            )

    agreement_between_sources = sum(agreements) / len(agreements) if agreements else 0.0

    # Final confidence
    confidence = (
        0.5 * avg_score +
        0.2 * agreement_between_sources +
        0.2 * score_consistency +
        0.1 * top_k_margin
    )

    return max(0.0, min(confidence, 1.0))