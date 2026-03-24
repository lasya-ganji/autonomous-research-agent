from typing import Dict, Any, List
from pydantic import ValidationError

from models.planner_models import PlanStep
from models.error_models import ErrorLog
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    
    # Extract input
    query: str = state.get("query")

    if not query:
        raise ValueError("Query missing in state")

    # Replan logic based on state
    is_replan: bool = state.get("replan_count", 0) > 0

    # Load prompt
    prompt_template = load_prompt(
        "planner_replan.txt" if is_replan else "planner_initial.txt"
    )

    prompt = prompt_template.format(
        query=query,
        previous_plan=state.get("research_plan", [])
    )

    # Call LLM
    response = call_llm(
        prompt=prompt,
        temperature=0.2,
        expect_json=True
    )

    # Validate response
    plan: List[PlanStep] = []

    if isinstance(response, list):
        for item in response:
            try:
                plan.append(PlanStep(**item))
            except ValidationError as e:
                state.setdefault("errors", []).append(
                    ErrorLog(
                        node="planner_node",
                        error_type="validation_error",
                        message=str(e)
                    ).model_dump()
                )
    else:
        # Invalid LLM output
        state.setdefault("errors", []).append(
            ErrorLog(
                node="planner_node",
                error_type="invalid_llm_response",
                message=str(response)
            ).model_dump()
        )

    # Fallback plan
    if not plan:
        plan = [
            PlanStep(
                step_id=1,
                question=query,
                priority=5
            )
        ]

    # Sort by priority
    plan = sorted(plan, key=lambda x: x.priority, reverse=True)

    # Update state
    # Store plan
    state["research_plan"] = [p.model_dump() for p in plan]

    # Reset search results for new plan
    state["search_results"] = {}

    # Track unresolved steps (IMPORTANT)
    state["unresolved_steps"] = list(range(len(plan)))

    # Reset counters
    state["search_retry_count"] = 0

    # Observability
    state["node_execution_count"] = state.get("node_execution_count", 0) + 1

    # Routing hint
    state["next_node"] = "searcher_node"

    return state