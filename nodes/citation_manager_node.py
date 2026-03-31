from models.state import ResearchState
from models.enums import CitationStatus

from services.citation.citation_service import build_citation

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time


@trace_node("citation_manager_node")
def citation_manager_node(state: ResearchState) -> ResearchState:

    print("Citation Manager Node")

    start_time = time.time()

    input_data = {
        "num_steps": len(state.search_results),
        "existing_citations": len(state.citations)
    }

    seen_ids = set()

    # COLLECT FROM SEARCH RESULTS
    
    for results in state.search_results.values():
        for r in results:

            citation_id = getattr(r, "citation_id", None)

            if not citation_id:
                continue  # 🚨 skip if searcher didn't assign

            if citation_id in seen_ids:
                continue

            seen_ids.add(citation_id)

            # build + validate
            citation = build_citation(r, citation_id)

            state.citations[citation_id] = citation

    # VALIDATION AGAINST SYNTHESIS

    broken_ids = set()

    if state.synthesis:
        for claim in state.synthesis.claims:
            for cid in claim.citations:

                if cid not in state.citations:
                    broken_ids.add(cid)

    # LOGGING

    num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
    num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
    num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

    state.node_logs["citation"] = {
        "num_citations": len(state.citations),
        "num_valid": num_valid,
        "num_broken": num_broken,
        "num_stale": num_stale,
        "missing_from_synthesis": list(broken_ids)
    }

    output_data = {
        "num_citations": len(state.citations),
        "valid": num_valid,
        "broken": num_broken,
        "stale": num_stale,
        "missing": len(broken_ids)
    }

    log_node_execution("citation", input_data, output_data, start_time)

    return state