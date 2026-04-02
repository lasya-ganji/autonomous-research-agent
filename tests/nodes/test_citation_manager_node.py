from nodes.citation_manager_node import citation_manager_node
from models.state import ResearchState
from models.citation_models import Citation
from datetime import datetime

def test_citation_manager_runs():
    state = ResearchState(query="AI")
    state.citations = {
    "c1": Citation(
        citation_id="c1",
        url="http://test.com",
        title="Test",
        date_accessed=datetime.now().isoformat(),
        quality_score=0.8,
        status="valid"
    )
}
    result = citation_manager_node(state)

    assert result.citations is not None