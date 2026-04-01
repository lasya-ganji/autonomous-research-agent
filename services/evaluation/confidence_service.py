from typing import List
from models.search_models import SearchResult


TOP_K = 5


def compute_confidence(results: List[SearchResult]) -> float:
    """
    PRD-aligned confidence calculation:
    - average_score
    - agreement_between_sources
    - score_consistency
    - top_k_margin
    """

    if not results:
        return 0.0

    # Step 1: Top-K selection
    results = sorted(results, key=lambda x: x.quality_score, reverse=True)[:TOP_K]
    scores = [r.quality_score for r in results]

    # Step 2: Average score
    avg_score = sum(scores) / len(scores)

    # Step 3: Score consistency (normalized)
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    score_consistency = max(0.0, 1 - variance)  # already normalized since scores ∈ [0,1]

    # Step 4: Top-k margin (normalized)
    if len(scores) > 1:
        top_k_margin = scores[0] - scores[1]
    else:
        top_k_margin = scores[0]

    top_k_margin = max(0.0, min(top_k_margin, 1.0))

    # Step 5: Agreement (better approximation)
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
                text_overlap(
                    results[i].snippet or results[i].title,
                    results[j].snippet or results[j].title
                )
            )

    agreement_between_sources = (
        sum(agreements) / len(agreements)
        if agreements else 0.5  # fallback instead of 0
    )

    # Step 6: Weighted confidence
    confidence = (
        0.35 * avg_score +
        0.25 * agreement_between_sources +
        0.20 * score_consistency +
        0.20 * top_k_margin
    )

    # Step 7: Mild smoothing (temporary until embeddings)
    confidence = min(1.0, confidence + 0.05)

    return max(0.0, confidence)