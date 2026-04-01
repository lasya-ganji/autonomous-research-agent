from typing import List
from models.search_models import SearchResult
from retrieval.embedding_service import get_embedding
import numpy as np



TOP_K = 5


def compute_confidence(results: List[SearchResult]) -> float:
    """
    FINAL improved confidence calculation:
    - avg_score (strong signal)
    - consistency (variance)
    - top_k_margin (normalized)
    - agreement_between_sources (stable)
    - top_score boost (controlled)
    - smoothing (NEW)
    """

    if not results:
        return 0.0

    # ---------------------------
    # 🔹 Top-k selection
    # ---------------------------
    results = sorted(results, key=lambda x: x.quality_score, reverse=True)[:TOP_K]

    scores = [r.quality_score for r in results]

    # ---------------------------
    # 1. Average score
    # ---------------------------
    avg_score = sum(scores) / len(scores)

    # ---------------------------
    # 2. Consistency (variance)
    # ---------------------------
    variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
    score_consistency = max(0.0, 1 - variance)

    # ---------------------------
    # 3. Top-k margin (FIXED)
    # ---------------------------
    if len(scores) > 1:
        top_k_margin = scores[0] - scores[1]
    else:
        top_k_margin = scores[0]

    # normalize
    top_k_margin = max(0.0, min(top_k_margin, 1.0))

# ---------------------------
# 4. Agreement (SEMANTIC - UPDATED)
# ---------------------------

    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0

        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0

        return float(np.dot(a, b) / denom)


    agreements = []

# Step 1: Prepare texts
    texts = [
        (r.title or "") + " " + (r.snippet or "")
        for r in results
    ]

# Step 2: Generate embeddings
    embeddings = []
    for text in texts:
        emb = get_embedding(text)

        if emb is not None:
            embeddings.append(np.array(emb, dtype=np.float32))
        else:
            embeddings.append(None)

# Step 3: Compute pairwise similarity
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb_i = embeddings[i]
            emb_j = embeddings[j]

            if emb_i is None or emb_j is None:
                continue

            sim = cosine_similarity(emb_i, emb_j)
            agreements.append(sim)

# Step 4: Final score (keep your fallback)
    agreement_between_sources = (
        sum(agreements) / len(agreements)
        if agreements else 0.5
    )

    # ---------------------------
    # 5. Top score boost (CONTROLLED)
    # ---------------------------
    top_score = scores[0]

    if top_score > 0.75:
        top_score_boost = 0.10
    elif top_score > 0.6:
        top_score_boost = 0.05
    else:
        top_score_boost = 0.0

    # ---------------------------
    # 6. Base confidence (BALANCED)
    # ---------------------------
    confidence = (
        0.45 * avg_score +
        0.20 * agreement_between_sources +
        0.20 * score_consistency +
        0.15 * top_k_margin
    )

    # ---------------------------
    # 7. Apply boost
    # ---------------------------
    confidence += top_score_boost

    # ---------------------------
    # 🔥 8. Smoothing (VERY IMPORTANT)
    # ---------------------------
    if confidence > 0.5:
        confidence += 0.03

    return max(0.0, confidence)