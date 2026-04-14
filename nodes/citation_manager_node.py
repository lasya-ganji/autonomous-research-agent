from datetime import datetime, timezone
from typing import Dict, List, Set

from models.state import ResearchState
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from services.citation.citation_service import validate_url
from services.retrieval.embedding_service import get_embedding
from services.retrieval.dedup_service import cosine_similarity

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames

from config.constants.citation_constants import (
    SIMILARITY_THRESHOLD,
    VERIFICATION_THRESHOLD,
    MIN_REQUIRED_CITATIONS
)


def normalize_citation_id(cid: str) -> str:
    cid = str(cid).strip()
    if not cid.startswith("["):
        cid = f"[{cid}]"
    return cid


def text_overlap(a: str, b: str) -> float:
    a_words = set((a or "").lower().split())
    b_words = set((b or "").lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


_embedding_cache: Dict[str, List[float]] = {}


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


def compute_similarity_score(claim_text: str, chunk: str) -> float:
    overlap = text_overlap(claim_text, chunk)

    emb1 = get_cached_embedding(claim_text)
    emb2 = get_cached_embedding(chunk)

    if emb1 is None or emb2 is None:
        return overlap

    semantic = cosine_similarity(emb1, emb2)
    return max(overlap, semantic)


@trace_node(NodeNames.CITATION_MANAGER)
def citation_manager_node(state: ResearchState) -> ResearchState:

    # -------------------------------
    # SAFETY INIT
    # -------------------------------
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    if not hasattr(state, "failure_counts") or state.failure_counts is None:
        state.failure_counts = {}

    state.failure_counts.setdefault("search_failures", 0)
    state.failure_counts.setdefault("parsing_failures", 0)
    state.failure_counts.setdefault("low_confidence", 0)
    state.failure_counts.setdefault("citation_failures", 0)

    if state.citations is None:
        state.errors.append(
            ErrorLog(
                node="citation_manager_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.ERROR,
                error_type=ErrorTypeEnum.parsing_error,
                message="Citations was None → reset",
            )
        )
        state.citations = {}

    if state.citation_chunks is None:
        state.citation_chunks = {}

    input_data = {
        "num_citations": len(state.citations),
        "num_claims": len(state.synthesis.claims) if state.synthesis else 0
    }

    # -------------------------------
    # URL VALIDATION
    # -------------------------------
    for cid, citation in state.citations.items():
        try:
            if citation.status != CitationStatus.valid:
                citation.status = validate_url(str(citation.url))
        except Exception as e:
            citation.status = CitationStatus.broken

            state.errors.append(
                ErrorLog(
                    node="citation_manager_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message=f"URL validation failed for {cid}: {str(e)}",
                )
            )

    used_ids: Set[str] = set()
    dropped_citations = 0
    claim_debug = {}

    # -------------------------------
    # CLAIM VALIDATION
    # -------------------------------
    if state.synthesis and state.synthesis.claims:

        for claim in state.synthesis.claims:

            cleaned_ids = []
            citation_scores = {}
            hallucinated = []

            for cid in claim.citation_ids:
                cid = normalize_citation_id(cid)
                citation_obj = state.citations.get(cid)

                if not citation_obj or citation_obj.status != CitationStatus.valid:
                    hallucinated.append(cid)
                    continue

                chunks = state.citation_chunks.get(cid, [])
                if not chunks:
                    hallucinated.append(cid)
                    continue

                best_score = 0.0

                for chunk in chunks:
                    score = compute_similarity_score(claim.text, chunk)
                    if score > best_score:
                        best_score = score

                if best_score > SIMILARITY_THRESHOLD:
                    cleaned_ids.append(cid)
                    citation_scores[cid] = best_score
                else:
                    hallucinated.append(cid)
                    dropped_citations += 1

            if citation_scores:
                avg_score = sum(citation_scores.values()) / len(citation_scores)

                claim.citation_confidence = avg_score
                claim.citation_ids = cleaned_ids
                claim.citation_score_map = citation_scores
                claim.hallucinated_citations = hallucinated

                claim.verified = avg_score > VERIFICATION_THRESHOLD

                if claim.verified:
                    used_ids.update(cleaned_ids)
                else:
                    state.failure_counts["citation_failures"] += 1

            else:
                claim.verified = False
                claim.citation_ids = []
                claim.citation_confidence = 0.0
                claim.citation_score_map = {}
                claim.hallucinated_citations = hallucinated
                state.failure_counts["citation_failures"] += 1

            claim_debug[claim.text[:80]] = {
                "citations": len(cleaned_ids),
                "confidence": round(claim.citation_confidence, 3),
                "verified": claim.verified
            }

    state.used_citation_ids = used_ids

    # -------------------------------
    # PARTIAL PROPAGATION
    # -------------------------------
    if state.synthesis is not None:
        if len(used_ids) < MIN_REQUIRED_CITATIONS or state.is_partial:
            state.synthesis.partial = True

        if state.synthesis.partial:
            state.is_partial = True

    print(f"[CITATION] Used: {len(used_ids)} | Dropped: {dropped_citations}")

    # -------------------------------
    # METRICS
    # -------------------------------
    num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
    num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
    num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

    node_name = NodeNames.CITATION_MANAGER
    existing_log = state.node_logs.get(node_name, {})

    existing_log.update({
        "total_sources": len(state.citations),
        "valid": num_valid,
        "broken": num_broken,
        "stale": num_stale,
        "used": len(used_ids),
        "dropped": dropped_citations,
        "claim_debug": claim_debug,
        "failure_counts": dict(state.failure_counts),
        "errors_count": len(state.errors)
    })

    state.node_logs[node_name] = existing_log

    log_node_execution(
        "citation_manager_node",
        input_data,
        {
            "used_citations": len(used_ids),
            "valid": num_valid,
            "dropped": dropped_citations
        }
    )

    state.node_execution_count += 1
    return state