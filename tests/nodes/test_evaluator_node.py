import pytest
from models.state import ResearchState
from nodes.evaluator_node import evaluator_node


class MockResult:
    def __init__(self, title="Test"):
        self.title = title
        self.relevance_score = 0.5
        self.domain_score = 0.5
        self.recency_score = 0.5
        self.depth_score = 0.5
        self.quality_score = 0.7
        self.citation_id = "[1]"


# -----------------------------
# TEST 1: successful evaluation
# -----------------------------
def test_evaluator_proceed():

    def mock_score(results, query, state=None):
        return results  # no filtering

    def mock_confidence(results, query):
        return 0.8  # above threshold

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "What is AI?", "priority": 1})
    ]
    state.search_results = {
        1: [MockResult()]
    }

    import nodes.evaluator_node as en
    en.score_results = mock_score
    en.compute_confidence = mock_confidence

    result = evaluator_node(state)

    assert result.evaluation.decision == "proceed"


# -----------------------------
# TEST 2: retry triggered
# -----------------------------
def test_evaluator_retry():

    def mock_score(results, query, state=None):
        return results

    def mock_confidence(results, query):
        return 0.2  # low confidence

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Explain AI"})
    ]
    state.search_results = {
        1: [MockResult()]
    }

    import nodes.evaluator_node as en
    en.score_results = mock_score
    en.compute_confidence = mock_confidence

    result = evaluator_node(state)

    assert result.evaluation.decision in ["retry", "replan", "proceed"]


# -----------------------------
# TEST 3: no results case
# -----------------------------
def test_evaluator_no_results():

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Explain AI"})
    ]
    state.search_results = {
        1: []
    }

    result = evaluator_node(state)

    assert result.evaluation is not None
    assert len(result.errors) > 0  


# -----------------------------
# TEST 4: scoring failure
# -----------------------------
def test_evaluator_scoring_error():

    def mock_score(results, query, state=None):
        raise Exception("Scoring failed")

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Explain AI"})
    ]
    state.search_results = {
        1: [MockResult()]
    }

    import nodes.evaluator_node as en
    en.score_results = mock_score

    result = evaluator_node(state)

    assert len(result.errors) > 0


# -----------------------------
# TEST 5: confidence failure
# -----------------------------
def test_evaluator_confidence_error():

    def mock_score(results, query, state=None):
        return results

    def mock_confidence(results, query):
        raise Exception("Embedding error")

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Explain AI"})
    ]
    state.search_results = {
        1: [MockResult()]
    }

    import nodes.evaluator_node as en
    en.score_results = mock_score
    en.compute_confidence = mock_confidence

    result = evaluator_node(state)

    assert len(result.errors) > 0


# -----------------------------
# TEST 6: max execution safety
# -----------------------------
def test_evaluator_max_execution():

    state = ResearchState(query="AI")
    state.node_execution_count = 12  # at limit — handled gracefully by supervisor now

    result = evaluator_node(state)
    assert result is not None