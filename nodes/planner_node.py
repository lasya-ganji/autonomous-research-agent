import json
from datetime import datetime, timezone
from typing import List

from observability.tracing import trace_node

from config.constants.node_constants.node_names import NodeNames
from config.constants.node_constants.planner_constants import (
    MAX_PLAN_STEPS,
    MIN_QUESTION_LENGTH,
    MAX_PRIORITY,
    AVERAGE_OVERLAP_THRESHOLD
)
from config.constants.llm_constants import PLANNER_TEMPERATURE

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
                failure_reason=str(state.failure_reason or "Low quality results"),
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
        res = llm_fn(prompt=prompt, temperature=PLANNER_TEMPERATURE)

        if isinstance(res, dict) and res.get("error"):

            raw_type = res.get("error_type", "unknown_error")

            if raw_type == "api_error":
                error_type = ErrorTypeEnum.api_error
                severity = SeverityEnum.CRITICAL

            elif raw_type == "timeout_error":
                error_type = ErrorTypeEnum.timeout_error
                severity = SeverityEnum.ERROR

            elif raw_type == "network_error":
                error_type = ErrorTypeEnum.network_error
                severity = SeverityEnum.ERROR

            else:
                error_type = ErrorTypeEnum.system_error
                severity = SeverityEnum.ERROR

            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=severity,
                    error_type=error_type,
                    message=f"LLM failure: {res.get('error')}",
                )
            )

            if error_type == ErrorTypeEnum.api_error:
                state.is_partial = True
                state.node_execution_count += 1
                return state

            response = ""
            usage = {}

        else:
            response = res.get("content", "") if isinstance(res, dict) else res
            usage = res.get("usage", {}) if isinstance(res, dict) else {}

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        node_cost = calculate_cost(prompt_tokens, completion_tokens)

        state.total_tokens += usage.get("total_tokens", 0)
        state.total_cost += node_cost

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

        invalid_steps = 0

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except Exception:
                try:
                    start = response.find("[")
                    end = response.rfind("]") + 1
                    response = json.loads(response[start:end])
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


        def is_query_aligned(query: str, questions: list[str]) -> bool:
            query_words = set(query.lower().split())

            if len(query_words) == 0:
                return False

            overlap_scores = []

            for q in questions:
                q_words = set(q.lower().split())
                overlap = len(query_words & q_words) / max(len(query_words), 1)

                print(f"[PLANNER DEBUG] Question: {q}")
                print(f"[PLANNER DEBUG] Overlap: {overlap:.3f}")

                overlap_scores.append(overlap)

            avg_overlap = sum(overlap_scores) / len(overlap_scores)

            print(f"[PLANNER DEBUG] Avg overlap: {avg_overlap:.3f}")

            return avg_overlap > AVERAGE_OVERLAP_THRESHOLD

        if plan:
            questions = [step.question for step in plan]

            if not is_query_aligned(state.query, questions):

                print("[PLANNER DEBUG] Query NOT grounded → marking low confidence")

                state.failure_counts["low_confidence"] += 1

                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.low_confidence,
                        message="Generated plan not aligned with query"
                    )
                )

                state.failure_reason = "query_not_grounded"
                state.is_partial = True

        fallback_used = False

        if not plan:
            fallback_used = True

            state.failure_counts["low_confidence"] += 1

            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.low_confidence,
                    message="Fallback plan used — weak or invalid query"
                )
            )

            state.failure_reason = "weak_query"

            fallback_question = query.strip()
            if len(fallback_question) < MIN_QUESTION_LENGTH:
                fallback_question = f"Explain {fallback_question}"

            plan = [PlanStep(step_id=1, question=fallback_question, priority=1)]

            if fallback_used:
                state.is_partial = True

        print(f"[PLANNER DEBUG] Final plan steps: {len(plan)} | Replan: {is_replan} | Fallback: {fallback_used}")

        plan = sorted(plan, key=lambda x: x.priority)[:MAX_PLAN_STEPS]

        for idx, step in enumerate(plan, start=1):
            step.step_id = idx

        state.research_plan = plan

        if is_replan:
            state.search_results = {}

        existing_log = state.node_logs.get(NodeNames.PLANNER, {})

        existing_log.update({
            "num_steps": len(plan),
            "questions": [step.question for step in plan],
            "is_replan": is_replan,
            "invalid_plan_steps": invalid_steps,
            "fallback_used": fallback_used,
            "errors_count": len(state.errors),
            "node_tokens": usage.get("total_tokens", 0),
            "node_cost": round(node_cost, 4),
            "total_cost": round(state.total_cost, 4),
        })

        state.node_logs[NodeNames.PLANNER] = existing_log

        log_node_execution(
            "planner_node",
            {"query": query},
            {"num_steps": len(plan)}
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
                message=f"Planner crash: {str(e)}",
            )
        )

        fallback = state.query or "Provide a research plan."

        state.research_plan = [
            PlanStep(step_id=1, question=fallback, priority=1)
        ]

        state.node_execution_count += 1
        return state