import time
from functools import wraps


def trace_node(node_name: str):
    """
    Tracing decorator (PRD-aligned).

    Goals:
    - DO NOT overwrite node debug data
    - Preserve all node-level logs
    - Attach execution metadata safely
    - Keep duration for UI (not for debug overwrite)
    """

    def decorator(func):

        @wraps(func)
        def wrapper(state, *args, **kwargs):

            # Ensure state structure
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

                error_payload = {
                    "node": node_name,
                    "error": str(e),
                    "timestamp": time.time()
                }

                state.errors.append(error_payload)

                print(f"[TRACE ERROR] {node_name}: {str(e)}")
                raise

            finally:
                duration = round(time.time() - start_time, 3)

                print(f"[TRACE END] {node_name} ({duration}s)")

                existing_log = state.node_logs.get(node_name, {})

                existing_log["_trace"] = {
                    "duration_s": duration,
                    "status": status
                }

                state.node_logs[node_name] = existing_log

        return wrapper

    return decorator