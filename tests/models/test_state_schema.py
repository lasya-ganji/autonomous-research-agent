import pytest
from pydantic import ValidationError
from models.state import ResearchState
from models.planner_models import PlanStep
from models.citation_models import Citation
from models.enums import CitationStatus

def test_state_requires_query():
    with pytest.raises(ValidationError):
        ResearchState()


def test_state_default_values():
    state = ResearchState(query="test")
    assert state.research_plan == []
    assert state.citations == {}
    assert state.errors == []
    assert state.node_execution_count == 0
    assert state.total_cost == 0.0
    assert state.cost_limit == 2.0
    assert state.abort is False
    assert state.next_node is None
    assert state.is_partial is False


def test_state_node_execution_count_negative():
    with pytest.raises(ValidationError):
        ResearchState(query="test", node_execution_count=-1)


def test_state_search_retry_count_negative():
    with pytest.raises(ValidationError):
        ResearchState(query="test", search_retry_count=-1)


def test_state_replan_count_negative():
    with pytest.raises(ValidationError):
        ResearchState(query="test", replan_count=-1)


def test_planstep_question_too_short():
    with pytest.raises(ValidationError):
        PlanStep(step_id=1, question="hi", priority=1)


def test_planstep_priority_zero():
    with pytest.raises(ValidationError):
        PlanStep(step_id=1, question="What is AI?", priority=0)


def test_planstep_priority_too_high():
    with pytest.raises(ValidationError):
        PlanStep(step_id=1, question="What is AI?", priority=6)


def test_planstep_valid():
    step = PlanStep(step_id=1, question="What is AI?", priority=3)
    assert step.question == "What is AI?"
    assert step.priority == 3


def test_citation_quality_score_above_one():
    with pytest.raises(ValidationError):
        Citation(
            citation_id="[1]", title="T", url="http://example.com",
            quality_score=1.5, status=CitationStatus.valid, date_accessed="2025"
        )


def test_citation_quality_score_negative():
    with pytest.raises(ValidationError):
        Citation(
            citation_id="[1]", title="T", url="http://example.com",
            quality_score=-0.1, status=CitationStatus.valid, date_accessed="2025"
        )


def test_citation_invalid_url():
    with pytest.raises(ValidationError):
        Citation(
            citation_id="[1]", title="T", url="not-a-url",
            quality_score=0.5, status=CitationStatus.valid, date_accessed="2025"
        )
