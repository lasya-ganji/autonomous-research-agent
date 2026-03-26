import time


def trace_node(node_name: str):
    
    def decorator(func):
        def wrapper(state, *args, **kwargs):

            print(f"[TRACE] → {node_name}")   #entry log

            try:
                result = func(state, *args, **kwargs)

                print(f"[TRACE] ← {node_name}")    #exit log

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