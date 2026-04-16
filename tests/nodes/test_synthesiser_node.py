from models.state import ResearchState
from nodes.synthesiser_node import synthesiser_node


def test_synthesiser_no_results():
    state = ResearchState(query="AI")
    state.search_results = {}
    state.citations = {}

    result = synthesiser_node(state)

    assert result.synthesis.partial is True
    assert len(result.errors) > 0


def test_synthesiser_basic_flow():
    state = ResearchState(query="AI")

    # Fake minimal valid result object
    class R:
        def __init__(self):
            self.citation_id = "[1]"
            self.content = "This is a valid long content about AI " * 10
            self.snippet = "AI info"
            self.quality_score = 0.9

    state.search_results = {1: [R()]}

    from models.citation_models import Citation
    from models.enums import CitationStatus

    state.citations = {
        "[1]": Citation(
            citation_id="[1]",
            title="AI",
            url="http://test.com",
            quality_score=0.9,
            status=CitationStatus.valid,
            date_accessed="2025"
        )
    }

    result = synthesiser_node(state)

    assert result.synthesis is not None


def test_synthesiser_llm_failure():
    state = ResearchState(query="AI")

    state.search_results = {}
    state.citations = {}

    result = synthesiser_node(state)

    assert len(result.errors) > 0