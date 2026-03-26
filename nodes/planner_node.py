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
    start_time = time.time()

    query: str = state.query

    if not query:
        raise ValueError("Query missing in state")

    is_replan: bool = state.replan_count > 0

    if is_replan:
        prompt_template = load_prompt("planner_replan.txt")
    else:
        prompt_template = load_prompt("planner_initial.txt")

    if is_replan:
        previous_plan = "\n".join(
            [f"{step.step_id}. {step.question}" for step in state.research_plan]
        )

        prompt = prompt_template.format(
            query=query,
            previous_plan=previous_plan,
            failure_reason=state.failure_reason or "Low quality results"
        )

    else:
        prompt = prompt_template.format(
            query=query,
            previous_plan="",
            failure_reason=""
        )

    response = call_llm(
        prompt=prompt,
        temperature=0,
        expect_json=True
    )

    print("\n[PLANNER RESPONSE]:", response)

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

    if not plan:
        plan = [PlanStep(step_id=1, question=query, priority=5)]

    plan = plan[:3]

    # sort first
    plan = sorted(plan, key=lambda x: x.priority)

    # assign step_ids after sorting
    for idx, step in enumerate(plan, start=1):
        step.step_id = idx

    state.research_plan = plan
    state.search_results = {}
    state.unresolved_steps = [step.step_id for step in plan]
    state.search_retry_count = 0

    state.node_execution_count += 1

    log_node_execution(
        node_name="planner_node",
        input_data=query,
        output_data=[step.model_dump() for step in plan],
        start_time=start_time
    )

    return state