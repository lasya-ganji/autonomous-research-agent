from models.state import ResearchState
from models.report_models import ReportModel

def reporter_node(state: ResearchState) -> ResearchState:
    print("Reporter Node")

    report = ReportModel(
        title="Research Report",
        sections=[claim.text for claim in state.synthesis.claims],
        citations=list(state.citations.keys()),
        metadata={
            "query": state.query,
            "timestamp": "2026-01-01",
            "model": "gpt-4o",
            "word_count": 100
        }
    )

    state.report = report
    return state