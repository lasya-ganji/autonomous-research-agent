import faiss
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

from models.cache_models import CacheModel
from config.constants import (
    CACHE_MAX_SIZE,
    CACHE_SIMILARITY_THRESHOLD,
)


class SemanticCache:
    def __init__(self, dim: int):
        self.dim = dim
        self.max_size = CACHE_MAX_SIZE
        self.similarity_threshold = CACHE_SIMILARITY_THRESHOLD

        self.index = faiss.IndexFlatIP(dim)
        self.cache: List[CacheModel] = []
        self.embeddings: List[np.ndarray] = []

    # Normalize vector
    def _normalize(self, vec: List[float]) -> np.ndarray:
        if vec is None:
            raise ValueError("Embedding vector is None")

        arr = np.array(vec).astype("float32")
        norm = np.linalg.norm(arr)
        return arr / norm if norm > 0 else arr

    # Expiry check
    def _is_expired(self, entry: CacheModel) -> bool:
        created = datetime.fromisoformat(entry.timestamp)
        expiry = created + timedelta(seconds=entry.ttl_seconds)
        return datetime.utcnow() > expiry

    def _update_access(self, entry: CacheModel):
        entry.last_accessed = datetime.utcnow().isoformat()

    # TTL logic
    def _compute_ttl(self, quality: float, confidence: float) -> int:
        base = 300  # 5 min

        if quality > 0.8 and confidence > 0.8:
            return base * 6
        elif confidence < 0.5:
            return base * 1
        return base * 3

    # LRU eviction
    def _evict_lru(self):
        if not self.cache:
            return

        lru_index = min(
            range(len(self.cache)),
            key=lambda i: datetime.fromisoformat(self.cache[i].last_accessed),
        )

        self.cache.pop(lru_index)
        self.embeddings.pop(lru_index)

        # rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        if self.embeddings:
            self.index.add(np.array(self.embeddings))

    # ADD 
    def add(
        self,
        query: str,
        embedding: List[float],
        result,
        quality_score: float,
        citation_confidence: float,
        is_failed: bool = False,
    ):
        # DO NOT STORE FAILED
        if is_failed:
            print("[CACHE SKIP] Failed result not stored")
            return

        emb = self._normalize(embedding)

        ttl = self._compute_ttl(quality_score, citation_confidence)

        try:
            entry = CacheModel(
                query_text=query,
                embedding=embedding,
                result=result,
                timestamp=datetime.utcnow().isoformat(),
                ttl_seconds=ttl,
                last_accessed=datetime.utcnow().isoformat(),
                quality_score=quality_score,
                citation_confidence=citation_confidence,
                is_failed=is_failed,
            )
        except Exception as e:
            print(f"[CACHE ERROR] Pydantic validation failed: {e}")
            return

        # Evict if full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache.append(entry)
        self.embeddings.append(emb)

        self.index.add(np.array([emb]).astype("float32"))

        print(f"[CACHE STORE] {query[:60]}...")

    #  SEARCH
    def search(self, query_embedding: List[float], top_k: int = 3) -> Optional[CacheModel]:
        if not self.cache:
            print("[CACHE EMPTY]")
            return None

        q = self._normalize(query_embedding)

        scores, indices = self.index.search(np.array([q]).astype("float32"), top_k)
        if self.index.ntotal == 0:
            print("[CACHE EMPTY INDEX]")
            return None

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            entry = self.cache[idx]

            # Skip bad entries
            if entry.is_failed or entry.quality_score < 0.5:
                continue

            # Skip expired
            if self._is_expired(entry):
                continue

            # Similarity check
            if score >= self.similarity_threshold:
                self._update_access(entry)

                # Boost frequently used cache
                entry.quality_score = min(1.0, entry.quality_score + 0.05)

                print(f"[CACHE HIT] Score={score:.3f}")

                return entry

        print("[CACHE MISS]")
        return None