from models.state import ResearchState
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from services.citation.citation_service import validate_url

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time
from datetime import datetime, timezone


def normalize_citation_id(cid: str) -> str:
    cid = str(cid).strip()
    if not cid.startswith("["):
        cid = f"[{cid}]"
    return cid


@trace_node("citation_manager_node")
def citation_manager_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    # ✅ SAFE DEFAULTS (FIX)
    input_data = {}
    output_data = {}

    try:
        # Safety init
        if state.errors is None:
            state.errors = []

        if state.citations is None:
            state.errors.append(
                ErrorLog(
                    node="citation_manager_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Citations field was None, reset to empty dict"
                )
            )
            state.citations = {}

        if state.node_logs is None:
            state.node_logs = {}

        input_data = {
            "num_steps": len(state.search_results or {}),
            "existing_citations": len(state.citations)
        }

        # 1. VALIDATE
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
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"URL validation failed for {cid}: {str(e)}"
                    )
                )

        # 2. SYNTHESIS ALIGNMENT
        broken_ids = set()
        used_ids = set()

        valid_ids_set = {
            cid for cid, c in state.citations.items()
            if c.status == CitationStatus.valid
        }

        if state.synthesis:
            for claim in state.synthesis.claims:
                try:
                    cleaned_ids = []

                    for cid in claim.citation_ids:
                        cid = normalize_citation_id(cid)

                        if cid not in state.citations:
                            broken_ids.add(cid)
                            continue

                        if cid not in valid_ids_set:
                            continue

                        cleaned_ids.append(cid)

                    if not cleaned_ids and valid_ids_set:
                        cleaned_ids = [next(iter(valid_ids_set))]

                    claim.citation_ids = cleaned_ids
                    used_ids.update(cleaned_ids)

                except Exception as e:
                    state.errors.append(
                        ErrorLog(
                            node="citation_manager_node",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.parsing_error,
                            message=f"Claim alignment failed: {str(e)}"
                        )
                    )

        state.used_citation_ids = set(used_ids)

        # 3. METRICS
        num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
        num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
        num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

        # 4. LOGGING
        state.node_logs["citation"] = {
            "num_citations": len(state.citations),
            "num_valid": num_valid,
            "num_broken": num_broken,
            "num_stale": num_stale,
            "missing_from_synthesis": list(broken_ids),
            "used_citations": list(state.used_citation_ids)
        }

        output_data = {
            "num_citations": len(state.citations),
            "valid": num_valid,
            "broken": num_broken,
            "stale": num_stale,
            "missing": len(broken_ids),
            "used": len(state.used_citation_ids)
        }

        print(
            f"[CITATION DEBUG] total={len(state.citations)} "
            f"valid={num_valid} used={len(state.used_citation_ids)} "
            f"missing={len(broken_ids)}"
        )

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="citation_manager_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}"
            )
        )

    # ✅ ALWAYS SAFE NOW
    log_node_execution("citation", input_data, output_data, start_time)

    state.node_execution_count += 1

    return state