import json
from datetime import datetime, timezone
from typing import List

from observability.tracing import trace_node

from config.constants.node_names import NodeNames
from config.constants.planner_constants import (
    MAX_PLAN_STEPS,
    MIN_QUESTION_LENGTH,
    MAX_PRIORITY
)

from models.error_models import ErrorLog
from models.enums import ErrorTypeEnum, SeverityEnum
from models.planner_models import PlanStep
from models.state import ResearchState

from services.system.cost_tracker import calculate_cost
from tools.llm_tool import call_llm

from utils.logger import log_node_execution
from utils.prompt_loader import load_prompt


def _init_state_safety(state: ResearchState):
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    if not hasattr(state, "failure_counts") or state.failure_counts is None:
        state.failure_counts = {
            "search_failures": 0,
            "parsing_failures": 0,   
            "low_confidence": 0,
        }


def _parse_plan_response(response, state: ResearchState):
    """
    Validates LLM output and converts it into structured PlanStep objects.
    Returns:
        plan: List[PlanStep]
        invalid_steps: number of unusable steps from LLM
    """
    plan: List[PlanStep] = []
    invalid_steps = 0

    if isinstance(response, list):
        for idx, item in enumerate(response):
            try:
                question = str(item.get("question", "")).strip()
                priority = item.get("priority", idx + 1)

                if len(question) < MIN_QUESTION_LENGTH:
                    raise ValueError("Question too short")

                try:
                    priority = int(priority)
                except Exception:
                    priority = idx + 1

                priority = max(1, min(priority, MAX_PRIORITY))

                plan.append(
                    PlanStep(
                        step_id=idx + 1,
                        question=question,
                        priority=priority,
                    )
                )

            except Exception as e:
                invalid_steps += 1

                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"Invalid plan step: {str(e)}",
                    )
                )
    else:
        invalid_steps += 1

        state.errors.append(
            ErrorLog(
                node="planner_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.ERROR,
                error_type=ErrorTypeEnum.parsing_error,
                message="Planner response is not a list",
            )
        )

    return plan, invalid_steps


@trace_node(NodeNames.PLANNER)
def planner_node(state: ResearchState, llm_fn=call_llm) -> ResearchState:

    try:
        _init_state_safety(state)

        if not state.query:
            raise ValueError("Query missing in state")

        query = state.query
        is_replan = state.replan_count > 0

        # -------------------------------
        # PROMPT GENERATION
        # -------------------------------
        prompt_template = load_prompt(
            "planner_replan.txt" if is_replan else "planner_initial.txt"
        )

        if is_replan:
            previous_plan = "\n".join(
                [f"{step.step_id}. {step.question}" for step in state.research_plan]
            )

            failure_context = {
                "reason": state.failure_reason or "Low quality results",
                "search_failures": state.failure_counts.get("search_failures", 0),
            }

            prompt = prompt_template.format(
                query=query,
                previous_plan=previous_plan,
                failure_reason=str(failure_context),
            )
        else:
            prompt = prompt_template.format(
                query=query,
                previous_plan="",
                failure_reason="",
            )

        # -------------------------------
        # LLM CALL
        # -------------------------------
        res = llm_fn(prompt=prompt, temperature=0)

        if isinstance(res, dict):
            response = res.get("content", "")
            usage = res.get("usage", {})
        else:
            response = res
            usage = {}

        # -------------------------------
        # COST TRACKING
        # -------------------------------
        node_tokens = usage.get("total_tokens", 0)

        node_cost = calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        
        state.total_tokens += usage.get("total_tokens", 0)
        state.total_cost += calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        print(f"[PLANNER COST] tokens={state.total_tokens} cost=₹{round(state.total_cost, 4)}")

        if state.total_cost > state.cost_limit:
            state.abort = True

            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.budget_exceeded,
                    message=f"Cost exceeded limit: ₹{state.total_cost}",
                )
            )

            state.node_execution_count += 1
            return state

        # -------------------------------
        # RESPONSE PARSING + VALIDATION
        # -------------------------------
        invalid_steps = 0

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except Exception as e:
                invalid_steps += 1

                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.ERROR,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"JSON parsing failed: {str(e)}",
                    )
                )
                response = []

        plan, step_failures = _parse_plan_response(response, state)
        invalid_steps += step_failures

        # -------------------------------
        # FALLBACK HANDLING
        # -------------------------------
        fallback_used = False

        if not plan:
            fallback_used = True

            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Empty plan from LLM, fallback used",
                )
            )

            fallback_question = query.strip()
            if len(fallback_question) < MIN_QUESTION_LENGTH:
                fallback_question = f"Explain {fallback_question}"

            plan = [PlanStep(step_id=1, question=fallback_question, priority=1)]

        # -------------------------------
        # SORT + LIMIT
        # -------------------------------
        plan = sorted(plan, key=lambda x: x.priority)[:MAX_PLAN_STEPS]

        for idx, step in enumerate(plan, start=1):
            step.step_id = idx

        # -------------------------------
        # DEBUG (TERMINAL ONLY)
        # -------------------------------
        print(f"[PLANNER] Steps: {len(plan)} | Replan: {is_replan}")
        if fallback_used:
            print("[PLANNER WARNING] Fallback used")

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        state.research_plan = plan

        if is_replan:
            state.search_results = {}

        # -------------------------------
        # UI DEBUG (CLEAN + USEFUL)
        # -------------------------------
        existing_log = state.node_logs.get(NodeNames.PLANNER, {})

        existing_log.update({
            "num_steps": len(plan),
            "questions": [step.question for step in plan],
            "is_replan": is_replan,
            "reason": state.failure_reason if is_replan else "initial_plan",
            "invalid_plan_steps": invalid_steps, 
            "fallback_used": fallback_used,
            "errors_count": len(state.errors),
            "node_tokens": node_tokens,
            "node_cost": round(node_cost, 4),
            "total_cost": round(state.total_cost, 4),
        })

        state.node_logs[NodeNames.PLANNER] = existing_log

        log_node_execution(
            "planner_node",
            {
                "query": query,
                "is_replan": is_replan,
            },
            {
                "num_steps": len(plan),
                "cost": round(state.total_cost, 4),
            },
        )

        state.node_execution_count += 1
        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="planner_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=f"Unhandled planner error: {str(e)}",
            )
        )

        fallback_question = (
            state.query.strip() if state.query else "Provide a research plan."
        )

        state.research_plan = [
            PlanStep(step_id=1, question=fallback_question, priority=1)
        ]

        existing_log = state.node_logs.get(NodeNames.PLANNER, {})
        existing_log.update({
            "num_steps": 1,
            "questions": [fallback_question],
            "fallback_used": True,
            "errors_count": len(state.errors),
            "reason": "planner_exception",
        })

        state.node_logs[NodeNames.PLANNER] = existing_log

        log_node_execution(
            "planner_node",
            {"fallback": True},
            [state.research_plan[0].model_dump()],
        )

        state.node_execution_count += 1
        return state