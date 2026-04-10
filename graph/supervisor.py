import time
from datetime import datetime, timezone

from models.state import ResearchState
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames


MAX_NODE_EXECUTIONS = 12
MAX_SEARCH_FAILURES = 2


def _init_state_safety(state: ResearchState):
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    # Failure tracking (non-breaking addition)
    if not hasattr(state, "failure_counts") or state.failure_counts is None:
        state.failure_counts = {
            "search_failures": 0,
            "parsing_failures": 0,
            "low_confidence": 0,
        }

    # Partial output tracking 
    if not hasattr(state, "is_partial"):
        state.is_partial = False


def _should_finalize_partial(state: ResearchState) -> bool:
    """
    Centralized partial handling (PRD requirement)
    """
    if state.abort:
        return True

    if state.node_execution_count >= MAX_NODE_EXECUTIONS:
        return True

    if state.failure_counts.get("search_failures", 0) > MAX_SEARCH_FAILURES:
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

            # Proper error classification
            if state.abort:
                error_type = ErrorTypeEnum.budget_exceeded
                message = "Execution aborted due to cost limit"
            elif state.node_execution_count >= MAX_NODE_EXECUTIONS:
                error_type = ErrorTypeEnum.loop_limit
                message = "Max node execution limit reached"
            else:
                error_type = ErrorTypeEnum.low_confidence
                message = "Too many failed retrieval attempts"

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
                "next_node": state.next_node,
                "execution_count": state.node_execution_count,
                "retry_count": state.search_retry_count,
                "replan_count": state.replan_count,
                "abort": state.abort,
                "is_partial": True,
            }

            log_node_execution(
                "supervisor_node",
                {"decision": decision},
                {"next_node": state.next_node, "partial": True},
            )

            state.node_execution_count += 1
            return state

        # -------------------------------
        # NORMAL ROUTING
        # -------------------------------

        # FIRST RUN → planner
        if not state.evaluation:
            state.next_node = "planner"

        else:
            if decision == "proceed":
                state.next_node = "synthesiser"

            elif decision == "retry":
                state.next_node = "searcher"

            elif decision == "replan":
                state.next_node = "planner"

            else:
                # Safe fallback (no silent behavior)
                state.errors.append(
                    ErrorLog(
                        node="supervisor_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.ERROR,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"Unknown evaluator decision: {decision}",
                    )
                )
                state.next_node = "reporter"

        # -------------------------------
        # OBSERVABILITY
        # -------------------------------
        state.node_logs[NodeNames.SUPERVISOR] = {
            "decision": decision,
            "next_node": state.next_node,
            "execution_count": state.node_execution_count,
            "retry_count": state.search_retry_count,
            "replan_count": state.replan_count,
            "abort": state.abort,
            "is_partial": state.is_partial,
        }

        log_node_execution(
            "supervisor_node",
            {"decision": decision},
            {"next_node": state.next_node},
        )

        state.node_execution_count += 1
        return state

    except Exception as e:
        # SYSTEM FAILURE → SAFE PARTIAL EXIT
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
            {"next_node": "reporter", "partial": True},
        )

        return state