from nodes.evaluator_node import evaluator_node
from models.state import ResearchState

def test_evaluator_returns_decision():
    state = ResearchState(query="AI")
    state.search_results = {1: ["data"]}

    result = evaluator_node(state)

    assert result.evaluation.decision in ["proceed", "retry", "replan"]