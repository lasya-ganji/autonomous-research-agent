from unittest.mock import patch
from models.state import ResearchState
from models.citation_models import Citation
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus
from nodes.citation_manager_node import citation_manager_node

PATCH_EMB = patch("nodes.citation_manager_node.get_cached_embedding", return_value=None)


def valid_citation():
    return Citation(
        citation_id="[1]", title="T", url="http://example.com",
        quality_score=0.9, status=CitationStatus.valid, date_accessed="2025"
    )


def make_state(claim_text, citation_ids, chunks):
    state = ResearchState(query="AI")
    state.citations = {"[1]": valid_citation()}
    state.citation_chunks = {"[1]": chunks} if chunks else {}
    state.synthesis = SynthesisModel(
        claims=[Claim(
            text=claim_text, citation_ids=citation_ids,
            confidence=0.8, verified=True, citation_confidence=1.0
        )],
        conflicts=[], partial=False
    )
    return state


def test_unknown_citation_id_is_hallucinated():
    """Citation ID [999] not in registry → hallucinated"""
    state = make_state("Some claim", ["[999]"], ["some content"])
    with PATCH_EMB:
        result = citation_manager_node(state)
    assert "[999]" in result.synthesis.claims[0].hallucinated_citations
    assert result.synthesis.claims[0].verified is False


def test_no_chunks_is_hallucinated():
    """Valid citation but no scraped chunks → hallucinated (no evidence)"""
    state = make_state("AI is growing", ["[1]"], None)
    with PATCH_EMB:
        result = citation_manager_node(state)
    assert "[1]" in result.synthesis.claims[0].hallucinated_citations


def test_zero_overlap_is_hallucinated():
    """
    Zero word overlap → text_overlap = 0.0
    Not > SIMILARITY_THRESHOLD (0.4) → hallucinated
    """
    state = make_state(
        "quantum physics semiconductors",
        ["[1]"],
        ["football sports entertainment broadcasting"]
    )
    with PATCH_EMB:
        result = citation_manager_node(state)
    claim = result.synthesis.claims[0]
    assert "[1]" in claim.hallucinated_citations
    assert claim.citation_confidence == 0.0


def test_high_overlap_is_verified():
    """
    intersection=6, union=8 → text_overlap = 0.75
    0.75 > SIMILARITY_THRESHOLD (0.4) → passes
    0.75 > VERIFICATION_THRESHOLD (0.5) → verified=True
    """
    state = make_state(
        "AI is powerful tool for research",
        ["[1]"],
        ["AI is powerful tool for research and automation"]
    )
    with PATCH_EMB:
        result = citation_manager_node(state)
    claim = result.synthesis.claims[0]
    assert claim.verified is True
    assert "[1]" in result.used_citation_ids
