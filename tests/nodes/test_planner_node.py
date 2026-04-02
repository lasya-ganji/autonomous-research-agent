import pytest
from nodes.planner_node import planner_node
from models.state import ResearchState

class MockLLM:
    def generate(self, prompt):
        return [
            {"question": "What is AI?", "priority": 1}
        ]

def test_planner_generates_plan():
    state = ResearchState(query="AI")

    result = planner_node(state, llm=MockLLM())

    assert result.research_plan is not None
    assert len(result.research_plan) == 1