import time
from functools import wraps
from datetime import datetime, timezone

from config.constants.node_constants.node_names import NodeNames
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum


def trace_node(node_name: str):

    def decorator(func):

        @wraps(func)
        def wrapper(state, *args, **kwargs):

            if getattr(state, "node_logs", None) is None:
                state.node_logs = {}

            if getattr(state, "errors", None) is None:
                state.errors = []

            start_time = time.time()

            print(f"[TRACE START] {node_name}")

            status = "success"

            try:
                result = func(state, *args, **kwargs)
                return result

            except Exception as e:
                status = "error"

                state.errors.append(
                    ErrorLog(
                        node=node_name,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.CRITICAL,
                        error_type=ErrorTypeEnum.system_error,
                        message=str(e),
                    )
                )

                print(f"[TRACE ERROR] {node_name}: {str(e)}")
                raise

            finally:
                duration = round(time.time() - start_time, 3)

                print(f"[TRACE END] {node_name} ({duration}s)")

                existing_log = state.node_logs.get(node_name, {})
                trace = existing_log.get("_trace", {})

                total_duration = trace.get("total_duration_s", 0) + duration
                run_count = trace.get("run_count", 0) + 1

                trace.update({
                    "total_duration_s": round(total_duration, 3),
                    "run_count": run_count,
                    "last_duration_s": duration,
                    "status": status
                })

                existing_log["_trace"] = trace
                state.node_logs[node_name] = existing_log

                if node_name != NodeNames.SUPERVISOR:

                    total_runs = 0

                    for node, data in state.node_logs.items():
                        if node == NodeNames.SUPERVISOR:
                            continue

                        node_trace = data.get("_trace", {})
                        total_runs += node_trace.get("run_count", 0)

                    state.node_execution_count = total_runs

                    name = node_name.value if hasattr(node_name, "value") else str(node_name)
                    print(f"[COUNT] {state.node_execution_count} → {name}")

        return wrapper

    return decorator