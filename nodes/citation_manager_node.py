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

    # SAFETY INIT
    if state.errors is None:
        state.errors = []

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

    if state.node_logs is None:
        state.node_logs = {}

    if state.citation_chunks is None:
        state.citation_chunks = {}


    input_data = {
        "num_steps": len(state.search_results or {}),
        "existing_citations": len(state.citations)
    }

    # 1. URL VALIDATION
    for cid, citation in state.citations.items():
        try:
            if citation.status != CitationStatus.valid:
                citation.status = validate_url(str(citation.url))

                if citation.status == CitationStatus.broken:
                    citation.status = validate_url(str(citation.url))

            print(f"[URL] {cid} status={citation.status}")

        except Exception as e:
            citation.status = CitationStatus.broken
            print(f"[URL ERROR] {cid}: {e}")

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
    all_scores = []

    # 2. CLAIM VALIDATION
    if state.synthesis and getattr(state.synthesis, "claims", None):
        for claim in state.synthesis.claims:

            print(f"\n[CLAIM] {claim.text}")

            cleaned_ids = []
            citation_scores = {}
            hallucinated = []

            for cid in claim.citation_ids:
                cid = normalize_citation_id(cid)

                print(f"  [CITATION] {cid}")

                citation_obj = state.citations.get(cid)

                if not citation_obj:
                    print("    skip: not found")
                    hallucinated.append(cid)
                    continue

                if citation_obj.status != CitationStatus.valid:
                    print(f"    skip: status={citation_obj.status}")
                    hallucinated.append(cid)
                    continue

                chunks = state.citation_chunks.get(cid, [])
                if not chunks:
                    print("    skip: no chunks")
                    hallucinated.append(cid)
                    continue

                best_score = 0.0

                for i, chunk in enumerate(chunks):
                    score = compute_similarity_score(claim.text, chunk)
                    print(f"    chunk[{i}] score={round(score, 3)}")

                    if score > best_score:
                        best_score = score

                print(f"    best_score={round(best_score, 3)}")
                all_scores.append(best_score)

                if best_score > 0.3:
                    print("    pass")
                    cleaned_ids.append(cid)
                    citation_scores[cid] = best_score
                else:
                    print("    fail")
                    hallucinated.append(cid)

            print(f"  cleaned_ids={cleaned_ids}")
            print(f"  scores={[round(s, 3) for s in citation_scores.values()]}")

            # CLAIM DECISION
            if citation_scores:
                avg_score = sum(citation_scores.values()) / len(citation_scores)

                print(f"  avg_score={round(avg_score, 3)}")

                claim.citation_confidence = avg_score
                claim.citation_ids = cleaned_ids
                claim.citation_score_map = citation_scores
                claim.hallucinated_citations = hallucinated

                if avg_score > 0.4:
                    claim.verified = True
                    used_ids.update(cleaned_ids)
                else:
                    claim.verified = False

                print(f"  verified={claim.verified}")

            else:
                print("  no valid citations")

                claim.verified = False
                claim.citation_ids = []
                claim.citation_confidence = 0.0
                claim.citation_score_map = {}
                claim.hallucinated_citations = hallucinated

    state.used_citation_ids = used_ids

    print(f"\n[ALL SCORES] {[round(s, 3) for s in all_scores]}")

    # 3. METRICS
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

    existing_log = state.node_logs.get("citation", {})
    existing_log.update(metrics)
    state.node_logs["citation"] = existing_log

    existing_log = state.node_logs.get(NodeNames.CITATION_MANAGER, {})
    existing_log.update(metrics)
    state.node_logs[NodeNames.CITATION_MANAGER] = existing_log

    output_data = {
        "num_citations": len(state.citations),
        "valid": num_valid,
        "broken": num_broken,
        "stale": num_stale,
        "used": len(used_ids)
    }

    print(
        f"[CITATION DEBUG] total={len(state.citations)} "
        f"valid={num_valid} used={len(used_ids)}"
    )

    log_node_execution("citation", input_data, output_data)

    return state