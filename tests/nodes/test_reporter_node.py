from nodes.reporter_node import reporter_node
from models.state import ResearchState

def test_reporter_creates_report():
    state = ResearchState(query="AI")
    state.synthesis = type("obj", (), {"claims": []})

    result = reporter_node(state)

    assert result.report is not None