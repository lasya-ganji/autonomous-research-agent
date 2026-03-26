from models.state import ResearchState
from tools.search_tool import search_tool
from services.evaluation.scoring_service import score_results
from utils.logger import log_node_execution
import time


def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")
    start_time = time.time()

    try:
        for step in sorted(state.research_plan, key=lambda x: x.priority):
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            results = search_tool(query)

            if not results:
                print(f"[SEARCHER NODE] No results for step {step_id}")
                state.failed_queries.append(query)
                continue

            results = score_results(results, query)

            state.search_results[step_id] = results

        state.unresolved_steps = [
            step.step_id for step in state.research_plan
            if step.step_id not in state.search_results
        ]

        if state.unresolved_steps:
            state.search_retry_count += 1

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")
        state.failed_queries.append(state.query)

    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()},
        start_time=start_time
    )

    state.node_execution_count += 1

    return state