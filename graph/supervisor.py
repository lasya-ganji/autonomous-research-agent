import time
from datetime import datetime, timezone

from models.state import ResearchState
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_constants.node_names import NodeNames

from config.constants.node_constants.supervisor_constants import (
    MAX_NODE_EXECUTIONS,
    MAX_SEARCH_FAILURES, T1, T2, T3
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
    return (
        state.abort or
        state.node_execution_count >= MAX_NODE_EXECUTIONS or
        state.failure_counts.get("search_failures", 0) >= MAX_SEARCH_FAILURES
    )


@trace_node(NodeNames.SUPERVISOR)
def supervisor_node(state: ResearchState) -> ResearchState:

    try:
        _init_state_safety(state)

        # -------------------------------
        # TIME TRACKING
        # -------------------------------
        now = time.time()
        if getattr(state, "start_time", None) is None:
            state.start_time = now

        state.elapsed_time = now - state.start_time

        # -------------------------------
        # SAFE EVALUATION ACCESS
        # -------------------------------
        decision = None
        confidence = 0.0

        if state.evaluation:
            decision = getattr(state.evaluation, "decision", None)

            # support both avg_confidence / confidence
            confidence = getattr(
                state.evaluation,
                "avg_confidence",
                getattr(state.evaluation, "confidence", 0.0)
            )

        # -------------------------------
        # HARD FAIL: API FAILURE 
        # -------------------------------
        if getattr(state, "api_failure", False):

            state.is_partial = True
            state.abort = True
            state.next_node = "reporter"

            # Avoid duplicate logging
            if not any(e.error_type == ErrorTypeEnum.api_error for e in state.errors):
                state.errors.append(
                    ErrorLog(
                        node="supervisor_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.CRITICAL,
                        error_type=ErrorTypeEnum.api_error,
                        message="API authentication failed — stopping execution"
                    )
                )

            state.node_logs[NodeNames.SUPERVISOR] = {
                "decision": "abort",
                "reason": "api_error",
                "partial_trigger": "api_failure",
                "next_node": state.next_node,
                "execution_count": state.node_execution_count,
                "retry_count": state.search_retry_count,
                "replan_count": state.replan_count,
                "abort": True,
                "is_partial": True,
                "errors_count": len(state.errors),
                "elapsed_time": round(state.elapsed_time, 2),
            }

            log_node_execution(
                "supervisor_node",
                {"decision": "abort"},
                {"next_node": state.next_node, "reason": "api_error"}
            )

            return state
        

        # -------------------------------
        # PARTIAL FINALIZATION
        # -------------------------------
        if _should_finalize_partial(state):
            state.is_partial = True
            state.next_node = "reporter"

            if state.abort:
                error_type = ErrorTypeEnum.budget_exceeded
                message = "Execution aborted due to cost limit"
                trigger = "cost_limit"

            elif state.node_execution_count >= MAX_NODE_EXECUTIONS:
                error_type = ErrorTypeEnum.loop_limit
                message = "Max execution limit reached"
                trigger = "execution_limit"

            else:
                error_type = ErrorTypeEnum.low_confidence
                message = "Too many failed retrieval attempts"
                trigger = "search_failure_threshold"

            if not any(e.error_type == error_type for e in state.errors):
                state.errors.append(
                    ErrorLog(
                        node="supervisor_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.CRITICAL,
                        error_type=error_type,
                        message=message
                    )
                )

            state.node_logs[NodeNames.SUPERVISOR] = {
                "decision": decision,
                "reason": message,
                "partial_trigger": trigger,
                "next_node": state.next_node,
                "execution_count": state.node_execution_count,
                "retry_count": state.search_retry_count,
                "replan_count": state.replan_count,
                "abort": state.abort,
                "is_partial": True,
                "errors_count": len(state.errors),
                "elapsed_time": round(state.elapsed_time, 2),
            }

            log_node_execution(
                "supervisor_node",
                {"decision": decision},
                {"next_node": state.next_node, "reason": message}
            )

            return state

        # -------------------------------
        # LATENCY-AWARE DECISION
        # -------------------------------
        latency_reason = None

        if state.elapsed_time < T1:
            threshold = 0.6
        elif state.elapsed_time < T2:
            threshold = 0.55
        else:
            threshold = 0.5

        if state.elapsed_time >= T3:
            decision = "forced_proceed"
            state.is_partial = True
            latency_reason = "latency_critical_forced_proceed"

        elif confidence >= threshold:
            decision = "proceed"
            latency_reason = "latency_adjusted_proceed"

        elif state.elapsed_time >= T2:
            decision = "forced_proceed"
            state.is_partial = True
            latency_reason = "latency_low_confidence_forced_proceed"

        # -------------------------------
        # ROUTING
        # -------------------------------
        if not state.evaluation:
            state.next_node = "planner"
            reason = "initial_run"

        elif decision == "proceed":
            state.next_node = "synthesiser"
            reason = latency_reason or "evaluation_passed"

        elif decision == "forced_proceed":
            state.next_node = "synthesiser"
            state.is_partial = True
            reason = latency_reason or "forced_proceed"

        elif decision == "retry":
            if state.elapsed_time >= T2:
                state.next_node = "synthesiser"
                state.is_partial = True
                reason = "latency_retry_blocked"
            else:
                state.next_node = "searcher"
                reason = "retry"

        elif decision == "replan":
            if state.elapsed_time >= T2:
                state.next_node = "synthesiser"
                state.is_partial = True
                reason = "latency_replan_blocked"
            else:
                state.next_node = "planner"
                reason = "replan"

        else:
            state.next_node = "reporter"
            reason = "fallback"

        # -------------------------------
        # LOGGING
        # -------------------------------
        state.node_logs[NodeNames.SUPERVISOR] = {
            "decision": decision,
            "reason": reason,
            "next_node": state.next_node,
            "execution_count": state.node_execution_count,
            "retry_count": state.search_retry_count,
            "replan_count": state.replan_count,
            "abort": state.abort,
            "is_partial": state.is_partial,
            "errors_count": len(state.errors),
            "elapsed_time": round(state.elapsed_time, 2),
        }

        log_node_execution(
            "supervisor_node",
            {"decision": decision},
            {"next_node": state.next_node, "reason": reason}
        )

        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="supervisor_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=f"Supervisor failure: {str(e)}"
            )
        )

        state.is_partial = True
        state.next_node = "reporter"

        log_node_execution(
            "supervisor_node",
            {},
            {"next_node": "reporter", "reason": "exception"}
        )

        return state