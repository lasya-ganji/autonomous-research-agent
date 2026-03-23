from models.state import ResearchState
from models.search_models import SearchResult

def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")

    # Dummy search result
    state.search_results = {
        1: [
            SearchResult(
                citation_id="1",
                url="https://example.com",
                title="Sample Result",
                snippet="Sample snippet",
                content="Sample content",
                quality_score=0.8,
                relevance_score=0.9,
                recency_score=0.7,
                domain_score=0.8,
                depth_score=0.6,
                rank=1
            )
        ]
    }

    return state