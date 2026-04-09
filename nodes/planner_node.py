import json
import time
from datetime import datetime, timezone
from typing import List

from observability.tracing import trace_node

from config.constants.node_names import NodeNames
from models.error_models import ErrorLog
from models.enums import ErrorTypeEnum, SeverityEnum
from models.planner_models import PlanStep
from models.state import ResearchState
from services.system.cost_tracker import calculate_cost
from tools.llm_tool import call_llm
from utils.logger import log_node_execution
from utils.prompt_loader import load_prompt


@trace_node(NodeNames.PLANNER)
def planner_node(state: ResearchState, llm_fn=call_llm) -> ResearchState:
    start_time = time.time()


    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")
    if not state.query:
        raise ValueError("Query missing in state")

    try:
        # Safety init
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

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
                failure_reason=state.failure_reason or "Low quality results",
            )
        else:
            prompt = prompt_template.format(
                query=query,
                previous_plan="",
                failure_reason="",
            )

        # LLM call (supports mock dict responses)
        res = llm_fn(prompt=prompt, temperature=0)
        if isinstance(res, dict):
            response = res.get("content", "")
            usage = res.get("usage", {})
        else:
            response = res
            usage = {}

        # Token & cost tracking
        state.total_tokens += usage.get("total_tokens", 0)
        state.total_cost += calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

        # Cost guardrail
        if state.total_cost > state.cost_limit:
            state.abort = True
            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"Cost exceeded limit: ₹{state.total_cost}",
                )
            )
            state.node_execution_count += 1
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
                        message=f"JSON parsing failed: {str(e)}",
                    )
                )
                response = []

        plan: List[PlanStep] = []

        if isinstance(response, list):
            for idx, item in enumerate(response):
                try:
                    question = str(item.get("question", "")).strip()
                    priority = item.get("priority", idx + 1)

                    if len(question) < 5:
                        raise ValueError("Question too short")

                    try:
                        priority = int(priority)
                    except Exception:
                        priority = idx + 1

                    priority = max(1, min(priority, 5))
                    plan.append(
                        PlanStep(
                            step_id=idx + 1,
                            question=question,
                            priority=priority,
                        )
                    )
                except Exception as e:
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
            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message=f"Response is not a list: {str(response)}",
                )
            )

        # Fallback if empty
        if not plan:
            state.errors.append(
                ErrorLog(
                    node="planner_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Planner returned empty plan, using fallback",
                )
            )
            fallback_question = query.strip()
            if len(fallback_question) < 5:
                fallback_question = f"Explain {fallback_question}"
            plan = [PlanStep(step_id=1, question=fallback_question, priority=1)]

        # Sort & limit to top 3
        plan = sorted(plan, key=lambda x: x.priority)[:3]
        for idx, step in enumerate(plan, start=1):
            step.step_id = idx

        state.research_plan = plan
        state.search_results = {}
        state.search_retry_count = 0

        state.node_logs[NodeNames.PLANNER] = {
            "num_steps": len(plan),
            "questions": [step.question for step in plan],
        }

        log_node_execution(
            "planner_node",
            query,
            [step.model_dump() for step in plan],
            start_time,
        )

        state.node_execution_count += 1
        return state

    except Exception as e:
        
        if state.errors is None:
            state.errors = []
        state.errors.append(
            ErrorLog(
                node="planner_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unhandled planner error: {str(e)}",
            )
        )

        fallback_question = state.query.strip() if state.query else "Provide a research plan."
        state.research_plan = [PlanStep(step_id=1, question=fallback_question, priority=1)]
        state.search_results = {}
        state.search_retry_count = 0
        state.node_logs[NodeNames.PLANNER] = {
            "num_steps": 1,
            "questions": [fallback_question],
        }

        log_node_execution(
            "planner_node",
            fallback_question,
            [state.research_plan[0].model_dump()],
            start_time,
        )

        state.node_execution_count += 1
        return state