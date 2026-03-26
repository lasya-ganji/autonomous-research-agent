from models.state import ResearchState
from tools.search_tool import search_tool
from utils.logger import log_node_execution
from observability.tracing import trace_node
import time

@trace_node("searcher_node")
def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")
    start_time = time.time()

    # execution safety
    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    print(f"[DEBUG] Total steps in plan: {len(state.research_plan)}")

    state.search_results = {}

    try:
        for step in state.research_plan:
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            results = search_tool(query)

            state.search_results[step_id] = results if results else []

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")

    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()},
        start_time=start_time
    )

    state.node_execution_count += 1

    return state