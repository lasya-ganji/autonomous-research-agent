import time


def trace_node(node_name: str):
    
    def decorator(func):
        def wrapper(state, *args, **kwargs):
            start = time.time()

            print(f"[TRACE] → {node_name}")

            try:
                result = func(state, *args, **kwargs)

                duration = round((time.time() - start) * 1000, 2)
                print(f"[TRACE] ← {node_name} ({duration} ms)")

                return result

            except Exception as e:
                print(f"[TRACE ERROR] {node_name}: {str(e)}")

                # Store error in state
                if not hasattr(state, "errors"):
                    state.errors = []

                state.errors.append({
                    "node": node_name,
                    "error": str(e),
                    "timestamp": time.time()
                })

                raise e

        return wrapper

    return decorator