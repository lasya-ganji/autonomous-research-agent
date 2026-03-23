from models.state import ResearchState
from models.planner_models import PlanStep

def planner_node(state: ResearchState) -> ResearchState:
    print("Planner Node")

    # Dummy plan (replace with LLM later)
    state.research_plan = [
        PlanStep(step_id=1, question=state.query, priority=1)
    ]

    return state