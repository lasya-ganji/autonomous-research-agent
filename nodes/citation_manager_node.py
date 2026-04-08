from datetime import datetime, timezone
import time
from typing import Dict, List, Set

from models.state import ResearchState
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog
from models.synthesis_models import Claim

from services.citation.citation_service import validate_url
from services.retrieval.embedding_service import get_embedding
from services.retrieval.dedup_service import cosine_similarity

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames


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
    start_time = time.time()
    input_data: Dict = {}
    output_data: Dict = {}

    try:
        # Safety init (feature/others)
        if state.errors is None:
            state.errors = []
        if state.citations is None:
            state.errors.append(
                ErrorLog(
                    node="citation_manager_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Citations field was None, reset to empty dict",
                )
            )
            state.citations = {}
        if state.node_logs is None:
            state.node_logs = {}

        input_data = {
            "num_steps": len(state.search_results or {}),
            "existing_citations": len(state.citations),
        }

        # 1) URL validation with simple retry (HEAD)
        for cid, citation in state.citations.items():
            try:
                if citation.status != CitationStatus.valid:
                    citation.status = validate_url(str(citation.url))
                    # simple retry once if broken
                    if citation.status == CitationStatus.broken:
                        citation.status = validate_url(str(citation.url))
            except Exception as e:
                citation.status = CitationStatus.broken
                state.errors.append(
                    ErrorLog(
                        node="citation_manager_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"URL validation failed for {cid}: {str(e)}",
                    )
                )

        used_ids: Set[str] = set()

        # 2) CLAIM validation (semantic similarity over synthesiser-selected chunks)
        if state.synthesis and getattr(state.synthesis, "claims", None):
            # Cache valid ids for fallbacks
            valid_ids_set = {cid for cid, c in state.citations.items() if c.status == CitationStatus.valid}

            for claim in state.synthesis.claims:
                # Candidate ids: prefer what synthesiser asked for; fallback to all valid citations
                raw_candidate_ids = getattr(claim, "citation_ids", None) or []
                if raw_candidate_ids:
                    candidate_ids = [normalize_citation_id(cid) for cid in raw_candidate_ids]
                else:
                    candidate_ids = list(valid_ids_set)

                cleaned_ids: List[str] = []
                citation_scores: Dict[str, float] = {}
                hallucinated: List[str] = []

                for cid in candidate_ids:
                    citation_obj = state.citations.get(cid)
                    if not citation_obj:
                        hallucinated.append(cid)
                        continue
                    if citation_obj.status != CitationStatus.valid:
                        hallucinated.append(cid)
                        continue

                    chunks = (state.citation_chunks or {}).get(cid, [])
                    if not chunks:
                        hallucinated.append(cid)
                        continue

                    best_score = 0.0
                    for chunk in chunks:
                        score = compute_similarity_score(claim.text, chunk)
                        if score > best_score:
                            best_score = score

                    if best_score > 0.25:
                        cleaned_ids.append(cid)
                        citation_scores[cid] = best_score
                    else:
                        hallucinated.append(cid)

                # If nothing cleaned but we have valid citations, keep at least a trace
                if not cleaned_ids and valid_ids_set:
                    cleaned_ids = [next(iter(valid_ids_set))]

                if citation_scores:
                    avg_score = sum(citation_scores.values()) / len(citation_scores)
                    claim.citation_confidence = avg_score
                    claim.citation_ids = cleaned_ids
                    claim.citation_score_map = citation_scores
                    claim.hallucinated_citations = hallucinated
                    claim.verified = avg_score > 0.4
                    if claim.verified:
                        used_ids.update(cleaned_ids)
                else:
                    claim.verified = False
                    # Preserve fallback cleaned_ids even when no semantic scores were computed.
                    claim.citation_ids = cleaned_ids or []
                    claim.citation_confidence = 0.0
                    claim.citation_score_map = {}
                    claim.hallucinated_citations = hallucinated

        state.used_citation_ids = used_ids

        # 3) Metrics + node logs
        num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
        num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
        num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

        metrics = {
            "num_citations": len(state.citations),
            "num_valid": num_valid,
            "num_broken": num_broken,
            "num_stale": num_stale,
            "used_citations": list(used_ids),
        }
        # Keep both keys: tests expect "citation"; UI uses NodeNames.CITATION_MANAGER.
        state.node_logs["citation"] = metrics
        state.node_logs[NodeNames.CITATION_MANAGER] = metrics

        output_data = {
            "num_citations": len(state.citations),
            "valid": num_valid,
            "broken": num_broken,
            "stale": num_stale,
            "used": len(used_ids),
        }

        log_node_execution("citation", input_data, output_data, start_time)
        state.node_execution_count += 1
        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="citation_manager_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}",
            )
        )
        output_data = output_data or {}
        log_node_execution("citation", input_data, output_data, start_time)
        state.node_execution_count += 1
        return state

