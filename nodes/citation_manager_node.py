from models.state import ResearchState
from models.citation_models import Citation
from observability.tracing import trace_node


@trace_node("citation_manager_node")
def citation_manager_node(state: ResearchState) -> ResearchState:
    print("Citation Manager Node")

    for step_results in state.search_results.values():
        for res in step_results:
            state.citations[res.citation_id] = Citation(
                citation_id=res.citation_id,
                url=res.url,
                title=res.title,
                author=None,
                date_accessed="2026-01-01",
                quality_score=res.quality_score
            )

    return state