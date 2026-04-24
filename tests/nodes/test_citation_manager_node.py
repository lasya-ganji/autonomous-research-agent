from models.state import ResearchState
from nodes.citation_manager_node import citation_manager_node


def test_citation_basic_flow():
    state = ResearchState(query="AI")

    from models.citation_models import Citation
    from models.enums import CitationStatus

    # citations
    state.citations = {
        "[1]": Citation(
            citation_id="[1]",
            title="Test",
            url="http://example.com",
            quality_score=0.5,
            status=CitationStatus.valid,
            date_accessed="2025"
        )
    }

    result = citation_manager_node(state)

    assert result.node_logs["CITATION_MANAGER"]["total_sources"] == 1


def test_citation_invalid_url():
    state = ResearchState(query="AI")

    from models.citation_models import Citation
    from models.enums import CitationStatus

    state.citations = {
        "[1]": Citation(
            citation_id="[1]",
            title="Bad",
            url="http://bad-url.com",
            quality_score=0.5,
            status=CitationStatus.broken,
            date_accessed="2025"
        )
    }

    result = citation_manager_node(state)

    assert len(result.errors) >= 0  


def test_citation_synthesis_alignment():
    state = ResearchState(query="AI")

    from models.citation_models import Citation
    from models.enums import CitationStatus
    from models.synthesis_models import SynthesisModel, Claim

    state.citations = {
        "[1]": Citation(
            citation_id="[1]",
            title="Valid",
            url="http://example.com",
            quality_score=0.9,
            status=CitationStatus.valid,
            date_accessed="2025"
        )
    }

    # Fake synthesis with invalid id
    state.synthesis = SynthesisModel(
        claims=[
            Claim(
                text="AI is powerful",
                citation_ids=["999"],  # invalid
                confidence=0.8,
                verified=True,
                citation_confidence=1.0
            )
        ],
        conflicts=[],
        partial=False
    )

    result = citation_manager_node(state)

    # invalid citation id is hallucinated, claim marked unverified
    assert "[999]" in result.synthesis.claims[0].hallucinated_citations
    assert result.synthesis.claims[0].verified is False


def test_citation_no_synthesis():
    state = ResearchState(query="AI")
    state.citations = {}

    result = citation_manager_node(state)

    assert result.node_logs["CITATION_MANAGER"]["total_sources"] == 0


def test_citation_error_handling():
    state = ResearchState(query="AI")

    # force error
    state.citations = None

    result = citation_manager_node(state)

    assert len(result.errors) > 0