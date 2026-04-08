import pytest
from models.state import ResearchState
from nodes.planner_node import planner_node


# ✅ 1. Normal case
def test_planner_generates_plan():

    def mock_llm(prompt, temperature=0):
        return [
            {"question": "What is DevOps?", "priority": 1},
            {"question": "Benefits of DevOps", "priority": 2}
        ]

    state = ResearchState(query="Explain DevOps")

    result = planner_node(state, llm_fn=mock_llm)

    assert len(result.research_plan) > 0
    assert result.errors == []


# ✅ 2. Empty response → fallback
def test_planner_fallback():

    def mock_llm(prompt, temperature=0):
        return []

    state = ResearchState(query="AI")

    result = planner_node(state, llm_fn=mock_llm)

    assert len(result.research_plan) == 1
    assert len(result.errors) > 0
    assert "fallback" in result.errors[0].message.lower()


# ✅ 3. Invalid JSON
def test_planner_json_error():

    def mock_llm(prompt, temperature=0):
        return "INVALID JSON"

    state = ResearchState(query="ML")

    result = planner_node(state, llm_fn=mock_llm)

    assert any("JSON parsing failed" in e.message for e in result.errors)


# ✅ 4. Invalid step
def test_planner_invalid_step():

    def mock_llm(prompt, temperature=0):
        return [{"question": "hi", "priority": 1}]

    state = ResearchState(query="AI")

    result = planner_node(state, llm_fn=mock_llm)

    assert len(result.errors) > 0
    assert any("Invalid plan step" in e.message for e in result.errors)


# ✅ 5. Missing query
def test_planner_missing_query():

    state = ResearchState(query="")

    with pytest.raises(Exception):
        planner_node(state)