from typing import List
from pydantic import ValidationError
from datetime import datetime, timezone

from models.state import ResearchState
from models.planner_models import PlanStep
from models.error_models import ErrorLog
from tools.llm_tool import call_llm

from utils.prompt_loader import load_prompt
from utils.logger import log_node_execution
import time

start_time = time.time()

def planner_node(state: ResearchState) -> ResearchState:

    # Extract input
    query: str = state.query

    if not query:
        raise ValueError("Query missing in state")

    # Replan logic
    is_replan: bool = state.replan_count > 0

    # Load prompt
    if is_replan:
        prompt_template = load_prompt("planner_replan.txt")
    else:
        prompt_template = load_prompt("planner_initial.txt")

    prompt = prompt_template.format(query=query)

    # Call LLM
    response = call_llm(
        prompt=prompt,
        temperature=0,   # deterministic (as per PRD)
        expect_json=True
    )

    print("\n[PLANNER RESPONSE]:", response)

    # Validate response
    plan: List[PlanStep] = []

    if isinstance(response, list):
        for item in response:
            try:
                plan.append(PlanStep(**item))
            except ValidationError as e:
                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        error_type="parsing_error",
                        message=str(e),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity="ERROR"
                    )
                )
    else:
        state.errors.append(
            ErrorLog(
                node="planner_node",
                error_type="parsing_error",
                message=str(response),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="ERROR"
            )
        )

    # Fallback plan (safety)
    if not plan:
        plan = [
            PlanStep(
                step_id=1,
                question=query,
                priority=5
            )
        ]

    # 🔹 Enforce max 3 steps (PRD requirement)
    plan = plan[:3]

    # 🔹 Ensure step_ids are consistent (1,2,3...)
    for idx, step in enumerate(plan, start=1):
        step.step_id = idx

    # Sort by priority (optional but fine)
    plan = sorted(plan, key=lambda x: x.priority, reverse=True)

    # Store plan
    state.research_plan = plan

    # Reset search results
    state.search_results = {}

    # 🔹 Track unresolved steps correctly (use step_ids, not index)
    state.unresolved_steps = [step.step_id for step in plan]

    # Reset retry counter
    state.search_retry_count = 0

    # Observability
    state.node_execution_count += 1

    log_node_execution(
    node_name="planner_node",
    input_data=query,
    output_data=[step.model_dump() for step in plan],
    start_time=start_time
    )
    
    return state