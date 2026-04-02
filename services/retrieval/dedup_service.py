import math
from urllib.parse import urlparse
from typing import List, Tuple
from collections import defaultdict


# Stage 1: URL Normalization
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


# Utility: Cosine Similarity
def cosine_similarity(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# Stage 2: URL Deduplication
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


# Stage 3: Heuristic (Domain + Title)
def dedup_by_domain_title(results, similarity_threshold=0.85):
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

                # simple token overlap
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


# Stage 4: Semantic Deduplication
def semantic_dedup(results, embedding_fn, top_k=10, threshold=0.90):
    
    # skip if embeddings not available
    if embedding_fn is None:
        return results

    if not results:
        return results

    top_results = results[:top_k]
    rest = results[top_k:]

    embeddings = []
    for r in top_results:
        text = f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}"
        emb = embedding_fn(text)
        embeddings.append(emb)

    keep = [True] * len(top_results)

    for i in range(len(top_results)):
        if not keep[i]:
            continue

        for j in range(i + 1, len(top_results)):
            if not keep[j]:
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])

            if sim > threshold:
                qi = getattr(top_results[i], "quality_score", 0.5)
                qj = getattr(top_results[j], "quality_score", 0.5)

                if qi >= qj:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    deduped = [r for r, k in zip(top_results, keep) if k]

    return deduped + rest


# MAIN PIPELINE
def deduplicate_pipeline(raw_results, embedding_fn=None):

    initial_count = len(raw_results)

    # Stage 1 + 2
    step1 = dedup_by_url(raw_results)
    after_url = len(step1)

    # Stage 3
    step2 = dedup_by_domain_title(step1)
    after_heuristic = len(step2)

    # Stage 4
    if embedding_fn:
        step3 = semantic_dedup(step2, embedding_fn)
        after_semantic = len(step3)
    else:
        step3 = step2
        after_semantic = after_heuristic

    # DEBUG LOG
    print(
        f"[DEDUP DEBUG] "
        f"raw={initial_count} → "
        f"url={after_url} → "
        f"heuristic={after_heuristic} → "
        f"semantic={after_semantic}"
    )

    return step3