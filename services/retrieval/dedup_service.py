from services.retrieval.embedding_service import get_embedding

import math
from urllib.parse import urlparse
from typing import List
from collections import defaultdict


# CONFIG 
SEMANTIC_DUP_THRESHOLD = 0.80
TITLE_SIM_THRESHOLD = 0.85
TOP_K_SEMANTIC = 10
QUALITY_DIFF_THRESHOLD = 0.1  
MIN_QUALITY_FOR_DEDUP = 0.6


# EMBEDDING CACHE 
_embedding_cache = {}


def get_cached_embedding(text: str):
    if not text:
        return None

    if text in _embedding_cache:
        return _embedding_cache[text]

    try:
        emb = get_embedding(text)
        _embedding_cache[text] = emb
        return emb
    except Exception:
        return None


# URL NORMALIZATION
def normalize_url(url: str) -> str:
    try:
        url = str(url)
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except:
        return str(url)


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""


# COSINE SIMILARITY
def cosine_similarity(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# STAGE 1: URL DEDUP
def dedup_by_url(raw_results):
    seen_urls = set()
    unique = []

    for r in raw_results:
        url = str(getattr(r, "url", ""))
        if not url:
            continue

        norm_url = normalize_url(url)

        if norm_url in seen_urls:
            continue

        seen_urls.add(norm_url)
        unique.append(r)

    return unique


# STAGE 2: DOMAIN + TITLE HEURISTIC
def dedup_by_domain_title(results, similarity_threshold=TITLE_SIM_THRESHOLD):
    grouped = defaultdict(list)

    for r in results:
        domain = get_domain(getattr(r, "url", ""))
        grouped[domain].append(r)

    filtered = []

    for domain, docs in grouped.items():
        kept = []

        for doc in docs:
            title = (getattr(doc, "title", "") or "").lower()

            is_duplicate = False

            for k in kept:
                k_title = (getattr(k, "title", "") or "").lower()

                overlap = len(set(title.split()) & set(k_title.split()))
                total = len(set(title.split()) | set(k_title.split()))
                score = overlap / total if total else 0

                if score > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(doc)

        filtered.extend(kept)

    return filtered


# STAGE 3: SEMANTIC DEDUP
def semantic_dedup(results, top_k=TOP_K_SEMANTIC, threshold=SEMANTIC_DUP_THRESHOLD):
    if not results:
        return results

    top_results = results[:top_k]
    rest = results[top_k:]

    # Generate embeddings
    embeddings = []
    for r in top_results:
        text = f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}".strip()
        emb = get_cached_embedding(text)
        embeddings.append(emb)

    keep = [True] * len(top_results)

    for i in range(len(top_results)):
        if not keep[i] or embeddings[i] is None:
            continue

        for j in range(i + 1, len(top_results)):
            if not keep[j] or embeddings[j] is None:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim > threshold:
                qi = getattr(top_results[i], "quality_score", 0.5)
                qj = getattr(top_results[j], "quality_score", 0.5)

                # Condition 1: similarity is high
                # Condition 2: quality scores are close (avoid removing unique useful docs)
                quality_close = abs(qi - qj) < QUALITY_DIFF_THRESHOLD

                # Condition 3: both are high-quality → likely duplicate
                both_high_quality = qi > MIN_QUALITY_FOR_DEDUP and qj > MIN_QUALITY_FOR_DEDUP

                if quality_close or both_high_quality:
                    # remove lower quality
                    if qi >= qj:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                # else: do NOT remove (important for edge_case / same_topic)

    deduped = [r for r, k in zip(top_results, keep) if k]

    return deduped + rest


# MAIN PIPELINE
def deduplicate_pipeline(raw_results):
    initial_count = len(raw_results)

    # Stage 1
    step1 = dedup_by_url(raw_results)
    after_url = len(step1)

    # Stage 2
    step2 = dedup_by_domain_title(step1)
    after_heuristic = len(step2)

    # Stage 3
    step3 = semantic_dedup(step2)
    after_semantic = len(step3)

    # DEBUG LOG
    print(
        f"[DEDUP DEBUG] "
        f"raw={initial_count} → "
        f"url={after_url} → "
        f"heuristic={after_heuristic} → "
        f"semantic={after_semantic}"
    )

    return step3