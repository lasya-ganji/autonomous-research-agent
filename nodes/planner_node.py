from typing import List
from datetime import datetime, timezone
import time
import json

from models.state import ResearchState
from models.planner_models import PlanStep
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from utils.logger import log_node_execution
from observability.tracing import trace_node

from services.system.cost_tracker import calculate_cost


@trace_node("planner_node")
def planner_node(state: ResearchState, llm_fn=call_llm) -> ResearchState:
    start_time = time.time()

    try:
        # Safety init
        if state.errors is None:
            state.errors = []

        if state.node_execution_count >= 12:
            raise Exception("Max node execution limit reached")

        query = state.query
        if not query:
            raise ValueError("Query missing in state")

        is_replan = state.replan_count > 0

        prompt_template = load_prompt(
            "planner_replan.txt" if is_replan else "planner_initial.txt"
        )

        # Build prompt
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

        # UPDATED LLM CALL
        res = llm_fn(prompt=prompt, temperature=0)

        # HANDLE BOTH REAL + MOCK
        if isinstance(res, dict):
            response = res.get("content", "")
            usage = res.get("usage", {})
        else:
            response = res   # mock returns directly
            usage = {}

        print("\n[PLANNER RESPONSE RAW]:", response)

        # TOKEN TRACKING
        state.total_tokens += usage.get("total_tokens", 0)

        # COST TRACKING
        cost = calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        state.total_cost += cost

        # COST GUARDRAIL
        if state.total_cost > state.cost_limit:
            state.abort = True

            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"Cost exceeded limit: ₹{state.total_cost}"
                )
            )

            return state

        # Parse JSON
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except Exception as e:
                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.ERROR,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"JSON parsing failed: {str(e)}"
                    )
                )
                response = []

        plan: List[PlanStep] = []

        # Process response
        if isinstance(response, list):
            for idx, item in enumerate(response):
                try:
                    question = str(item.get("question", "")).strip()
                    priority = item.get("priority", idx + 1)

                    if len(question) < 5:
                        raise ValueError("Question too short")

                    try:
                        priority = int(priority)
                    except:
                        priority = idx + 1

                    priority = max(1, min(priority, 5))

                    plan.append(
                        PlanStep(
                            step_id=idx + 1,
                            question=question,
                            priority=priority
                        )
                    )

                except Exception as e:
                    state.errors.append(
                        ErrorLog(
                            node="planner_node",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.parsing_error,
                            message=f"Invalid plan step: {str(e)}"
                        )
                    )

        else:
            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message=f"Response is not a list: {str(response)}"
                )
            )

        # Fallback
        if not plan:
            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Planner returned empty plan, using fallback"
                )
            )

            fallback_question = query.strip()

            if len(fallback_question) < 5:
                fallback_question = f"Explain {fallback_question}"

            plan = [
                PlanStep(
                    step_id=1,
                    question=fallback_question,
                    priority=1
                )
            ]

        # Sort and limit
        plan = sorted(plan, key=lambda x: x.priority)[:3]

        for idx, step in enumerate(plan, start=1):
            step.step_id = idx

        # Update state
        state.research_plan = plan
        state.search_results = {}
        state.search_retry_count = 0

        state.node_execution_count += 1

        # Logging
        log_node_execution(
            node_name="planner_node",
            input_data=query,
            output_data=[step.model_dump() for step in plan],
            start_time=start_time
        )

        if state.node_logs is None:
            state.node_logs = {}

        state.node_logs["planner"] = {
            "num_steps": len(plan),
            "questions": [step.question for step in plan]
        }

        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="planner_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unhandled planner error: {str(e)}"
            )
        )
        raise e