from datetime import datetime
import requests
import time

from models.state import ResearchState
from models.citation_models import Citation
from models.enums import CitationStatus

from observability.tracing import trace_node
from utils.logger import log_node_execution


@trace_node("citation_manager_node")
def citation_manager_node(state: ResearchState) -> ResearchState:
    """
    Collect, validate, deduplicate, and store citations from search results.
    """

    print("Citation Manager Node")

    # Logger: start
    start_time = time.time()

    input_data = {
        "num_steps": len(state.search_results),
        "existing_citations": len(state.citations)
    }

    # Step 1: Flatten results
    all_results = []
    for results in state.search_results.values():
        all_results.extend(results)

    if not all_results:
        state.node_logs["citation"] = {"num_citations": 0}

        log_node_execution(
            "citation",
            input_data,
            {"num_citations": 0},
            start_time
        )
        return state

    # Step 2: Deduplicate by URL
    seen_urls = set()

    for r in all_results:
        url = getattr(r, "url", "")
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        citation_id = getattr(r, "citation_id", None) or str(hash(url))
        title = getattr(r, "title", "Untitled")
        quality_score = getattr(r, "quality_score", 0.5)  


        # Step 3: URL validation (lightweight)

        try:
            resp = requests.head(url, timeout=3, allow_redirects=True)

            if resp.status_code == 200:
                status = CitationStatus.valid
            elif resp.status_code in (301, 302):
                status = CitationStatus.stale
            else:
                status = CitationStatus.broken

        except requests.RequestException:
            status = CitationStatus.broken


        # Step 4: Store citation

        state.citations[citation_id] = Citation(
            citation_id=citation_id,
            title=title,
            url=url,
            quality_score=quality_score,
            status=status,
            date_accessed=datetime.now().isoformat()
        )

    # Step 5: Structured logs (UI/debug)
    num_valid = sum(1 for c in state.citations.values() if c.status == CitationStatus.valid)
    num_broken = sum(1 for c in state.citations.values() if c.status == CitationStatus.broken)
    num_stale = sum(1 for c in state.citations.values() if c.status == CitationStatus.stale)

    state.node_logs["citation"] = {
        "num_citations": len(state.citations),
        "num_valid": num_valid,
        "num_broken": num_broken,
        "num_stale": num_stale
    }

    # Step 6: Logger (system)
    output_data = {
        "num_citations": len(state.citations),
        "valid": num_valid,
        "broken": num_broken,
        "stale": num_stale
    }

    log_node_execution("citation", input_data, output_data, start_time)

    return state