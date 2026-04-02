# tests/nodes/test_synthesiser_node.py
import pytest
from nodes.synthesiser_node import synthesiser_node
from models.state import ResearchState
from models.search_models import SearchResult

# Mock LLM that returns valid claims

class MockLLM:
    def generate(self, prompt):
        # synthesiser_node expects a dict with 'claims' and 'conflicts'
        return {
            "claims": [
                {
                    "text": "AI is useful in many fields",
                    "citation_ids": ["c1"],
                    "confidence": 0.9
                }
            ],
            "conflicts": []
        }

def test_synthesiser_generates_claims():
    state = ResearchState(query="AI")
    state.search_results = {
        1: [
            SearchResult(
                citation_id="c1",
                url="http://test.com",
                title="Test",
                snippet="AI is useful in many fields including education, healthcare, and research.",
                content="AI is useful in many fields including education, healthcare, and research, with applications in automation, prediction, and decision-making.",
                quality_score=0.9,
                relevance_score=0.9,
                recency_score=0.9,
                domain_score=0.9,
                depth_score=0.9,
                rank=1
            )
        ]
    }

    result = synthesiser_node(state, llm=MockLLM())

    assert result.synthesis is not None
    assert len(result.synthesis.claims) > 0
    claim = result.synthesis.claims[0]
    assert claim.text == "AI is useful in many fields"
    assert claim.citation_ids == ["c1"]