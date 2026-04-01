from models.state import ResearchState
from models.enums import CitationStatus

from services.citation.citation_service import validate_url

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time


@trace_node("citation_manager_node")
def citation_manager_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    input_data = {
        "num_steps": len(state.search_results),
        "existing_citations": len(state.citations)
    }

    # VALIDATE ALL CITATIONS
    for cid, citation in state.citations.items():
        try:
            citation.status = validate_url(str(citation.url))
        except Exception:
            citation.status = CitationStatus.broken

    # SYNTHESIS ALIGNMENT
    broken_ids = set()
    used_ids = set()

    if state.synthesis:
        for claim in state.synthesis.claims:
            for cid in claim.citation_ids:

                if cid not in state.citations:
                    broken_ids.add(cid)
                else:
                    used_ids.add(cid)

    state.used_citation_ids = used_ids

    # LOGGING
    num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
    num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
    num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

    state.node_logs["citation"] = {
        "num_citations": len(state.citations),
        "num_valid": num_valid,
        "num_broken": num_broken,
        "num_stale": num_stale,
        "missing_from_synthesis": list(broken_ids),
        "used_citations": list(used_ids)
    }

    output_data = {
        "num_citations": len(state.citations),
        "valid": num_valid,
        "broken": num_broken,
        "stale": num_stale,
        "missing": len(broken_ids),
        "used": len(used_ids)
    }

    log_node_execution("citation", input_data, output_data, start_time)

    return state