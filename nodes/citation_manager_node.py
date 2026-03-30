import requests
from datetime import datetime
from models.state import ResearchState
from models.citation_models import CitationModel
from models.enums import CitationStatus
from observability.tracing import trace_node



@trace_node("citation_node")
def citation_node(state: ResearchState) -> ResearchState:
    """
    Collects, validates, and stores citations in the agent state.
    """

    print("Citation Node")

    # Step 1: Extract raw citations from search results
    raw_citations = getattr(state, "search_results", [])  # List of dicts with 'title', 'url', 'quality_score'

    for rc in raw_citations:
        citation_id = rc.get("id") or str(hash(rc.get("url")))
        title = rc.get("title", "Untitled")
        url = rc.get("url", "")
        quality_score = rc.get("quality_score", 0.5)

        # Step 2: Validate the URL (simple HEAD request)
        try:
            resp = requests.head(url, timeout=5, allow_redirects=True)
            if resp.status_code == 200:
                status = CitationStatus.valid
            elif resp.status_code in [301, 302]:
                status = CitationStatus.stale
            else:
                status = CitationStatus.broken
        except requests.RequestException:
            status = CitationStatus.broken

        # Step 3: Store in ResearchState.citations
        state.citations[citation_id] = CitationModel(
            citation_id=citation_id,
            title=title,
            url=url,
            quality_score=quality_score,
            status=status,
            date_accessed=datetime.now().isoformat()
        )

    # Step 4: Logging
    state.node_logs["citation"] = {
        "num_citations_collected": len(raw_citations),
        "num_valid": sum(1 for c in state.citations.values() if c.status == CitationStatus.valid),
        "num_broken": sum(1 for c in state.citations.values() if c.status == CitationStatus.broken),
        "num_stale": sum(1 for c in state.citations.values() if c.status == CitationStatus.stale),
    }

    return state