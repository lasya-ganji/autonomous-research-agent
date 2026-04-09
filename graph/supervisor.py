import time
from datetime import datetime, timezone

from models.state import ResearchState
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from observability.tracing import trace_node
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames


@trace_node(NodeNames.SUPERVISOR)
def supervisor_node(state: ResearchState) -> ResearchState:
    start_time = time.time()

    try:
        # Safety init
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

        if state.abort:
            state.next_node = "reporter"

            state.errors.append(
                ErrorLog(
                    node="supervisor_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message="Execution aborted due to cost limit"
                )
            )

        elif state.node_execution_count >= 12:
            raise Exception("Max node execution limit reached")

        # FIRST RUN → go to planner
        elif not state.evaluation:
            state.next_node = "planner"

        else:
            decision = state.evaluation.decision

            #  CORE ROUTING LOGIC
            if decision == "proceed":
                state.next_node = "synthesiser"

            elif decision == "retry":
                state.next_node = "searcher"

            elif decision == "replan":
                state.next_node = "planner"

            else:
                # fallback safety
                state.next_node = "synthesiser"

        #  OBSERVABILITY LOGS
        state.node_logs[NodeNames.SUPERVISOR] = {
            "decision": getattr(state.evaluation, "decision", None),
            "next_node": state.next_node,
            "execution_count": state.node_execution_count,
            "retry_count": state.search_retry_count,
            "replan_count": state.replan_count,
            "abort": state.abort
        }

        log_node_execution(
            "supervisor_node",
            {
                "decision": getattr(state.evaluation, "decision", None)
            },
            {
                "next_node": state.next_node
            },
            start_time
        )

        state.node_execution_count += 1
        return state

    except Exception as e:
        
        state.errors.append(
            ErrorLog(
                node="supervisor_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.timeout,
                message=f"Supervisor failure: {str(e)}"
            )
        )

        
        state.next_node = "reporter"

        log_node_execution(
            "supervisor_node",
            {},
            {"next_node": "reporter"},
            start_time
        )

        state.node_execution_count += 1
        return state