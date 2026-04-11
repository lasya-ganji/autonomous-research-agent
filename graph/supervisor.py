import time
from datetime import datetime, timezone

from models.state import ResearchState
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames

from config.constants.supervisor_constants import (
    MAX_NODE_EXECUTIONS,
    MAX_SEARCH_FAILURES
)


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

    if not hasattr(state, "is_partial"):
        state.is_partial = False


def _should_finalize_partial(state: ResearchState) -> bool:
    if state.abort:
        return True

    if state.node_execution_count >= MAX_NODE_EXECUTIONS:
        return True

    if state.failure_counts.get("search_failures", 0) >= MAX_SEARCH_FAILURES:
        return True

    return False


@trace_node(NodeNames.SUPERVISOR)
def supervisor_node(state: ResearchState) -> ResearchState:

    try:
        _init_state_safety(state)

        decision = getattr(state.evaluation, "decision", None)

        # -------------------------------
        # CENTRAL PARTIAL FINALIZATION
        # -------------------------------
        if _should_finalize_partial(state):
            state.is_partial = True
            state.next_node = "reporter"

            if state.abort:
                error_type = ErrorTypeEnum.budget_exceeded
                message = "Execution aborted due to cost limit"
                partial_trigger = "cost_limit"

            elif state.node_execution_count >= MAX_NODE_EXECUTIONS:
                error_type = ErrorTypeEnum.loop_limit
                message = "Max node execution limit reached"
                partial_trigger = "execution_limit"

            else:
                error_type = ErrorTypeEnum.low_confidence
                message = "Too many failed retrieval attempts"
                partial_trigger = "search_failure_threshold"

            state.errors.append(
                ErrorLog(
                    node="supervisor_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=error_type,
                    message=message,
                )
            )

            state.node_logs[NodeNames.SUPERVISOR] = {
                "decision": decision,
                "reason": message,
                "partial_trigger": partial_trigger,
                "next_node": state.next_node,
                "execution_count": state.node_execution_count + 1,
                "retry_count": state.search_retry_count,
                "replan_count": state.replan_count,
                "abort": state.abort,
                "is_partial": True,
                "errors_count": len(state.errors),
            }

            log_node_execution(
                "supervisor_node",
                {"decision": decision},
                {
                    "next_node": state.next_node,
                    "partial": True,
                    "reason": message,
                },
            )

            state.node_execution_count += 1
            return state

        # -------------------------------
        # NORMAL ROUTING
        # -------------------------------
        if not state.evaluation:
            state.next_node = "planner"
            reason = "initial_run"

        else:
            if decision == "proceed":
                state.next_node = "synthesiser"
                reason = "evaluation_passed"

            elif decision == "forced_proceed":
                state.next_node = "synthesiser"
                reason = "low_confidence_forced_proceed"

            elif decision == "retry":
                state.next_node = "searcher"
                reason = "low_confidence_retry"

            elif decision == "replan":
                state.next_node = "planner"
                reason = "retry_exhausted_replan"

            else:
                state.errors.append(
                    ErrorLog(
                        node="supervisor_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.ERROR,
                        error_type=ErrorTypeEnum.system_error,
                        message=f"Unknown evaluator decision: {decision}",
                    )
                )
                state.next_node = "reporter"
                reason = "unknown_decision_fallback"

        # -------------------------------
        # OBSERVABILITY
        # -------------------------------
        state.node_logs[NodeNames.SUPERVISOR] = {
            "decision": decision,
            "reason": reason,
            "next_node": state.next_node,
            "execution_count": state.node_execution_count + 1,
            "retry_count": state.search_retry_count,
            "replan_count": state.replan_count,
            "abort": state.abort,
            "is_partial": state.is_partial,
            "errors_count": len(state.errors),
        }

        log_node_execution(
            "supervisor_node",
            {"decision": decision},
            {
                "next_node": state.next_node,
                "reason": reason,
            },
        )

        state.node_execution_count += 1
        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="supervisor_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=f"Supervisor failure: {str(e)}",
            )
        )

        state.is_partial = True
        state.next_node = "reporter"

        log_node_execution(
            "supervisor_node",
            {},
            {
                "next_node": "reporter",
                "partial": True,
                "reason": "supervisor_exception",
            },
        )

        state.node_execution_count += 1
        return state