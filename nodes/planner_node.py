from typing import List
from datetime import datetime, timezone
import time
import json

from models.state import ResearchState
from models.planner_models import PlanStep
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from utils.logger import log_node_execution


def planner_node(state: ResearchState) -> ResearchState:
    start_time = time.time()

    # execution safety
    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    query = state.query
    if not query:
        raise ValueError("Query missing in state")

    is_replan = state.replan_count > 0

    prompt_template = load_prompt(
        "planner_replan.txt" if is_replan else "planner_initial.txt"
    )

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

    response = call_llm(prompt=prompt, temperature=0)

    print("\n[PLANNER RESPONSE RAW]:", response)

    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception as e:
            print(f"[PLANNER ERROR] JSON parsing failed: {e}")
            response = []

    plan: List[PlanStep] = []

    if isinstance(response, list):
        print(f"[DEBUG] Raw response length: {len(response)}")

        for idx, item in enumerate(response):
            try:
                # safe extraction + cleaning
                question = str(item.get("question", "")).strip()
                priority = item.get("priority", idx + 1)

                # enforce schema constraints manually
                if len(question) < 5:
                    raise ValueError("Question too short")

                try:
                    priority = int(priority)
                except:
                    priority = idx + 1

                # clamp priority within allowed range
                priority = max(1, min(priority, 5))

                step = PlanStep(
                    step_id=idx + 1,
                    question=question,
                    priority=priority
                )

                plan.append(step)

            except Exception as e:
                print(f"[PLANNER ERROR] Skipping invalid item: {item}, Error: {e}")
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
        print("[PLANNER ERROR] Response is not a list")
        state.errors.append(
            ErrorLog(
                node="planner_node",
                error_type="parsing_error",
                message=str(response),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="ERROR"
            )
        )

    print(f"[DEBUG] Parsed steps count: {len(plan)}")

    # fallback
    if not plan:
        plan = [PlanStep(step_id=1, question=query, priority=1)]

    # sort and limit
    plan = sorted(plan, key=lambda x: x.priority)[:3]

    # reassign step ids
    for idx, step in enumerate(plan, start=1):
        step.step_id = idx

    state.research_plan = plan
    state.search_results = {}
    state.search_retry_count = 0

    print(f"[DEBUG] Planner steps stored: {len(state.research_plan)}")

    state.node_execution_count += 1

    log_node_execution(
        node_name="planner_node",
        input_data=query,
        output_data=[step.model_dump() for step in plan],
        start_time=start_time
    )

    return state