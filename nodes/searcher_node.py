from models.state import ResearchState
from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url
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
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            results = search_tool(query)

            if results:
                for i, r in enumerate(results):
                    if not r.url:
                        continue

                        # Priority-based scraping
                    if step.priority == 1 and i < 3:
                        content = scrape_url(r.url)

                    elif step.priority == 2 and i < 2:
                        content = scrape_url(r.url)

                    elif step.priority == 3 and i < 1:
                        content = scrape_url(r.url)

                    else:
                        content = None

                    if content:
                        r.content = content  

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
    if state.node_logs is None:
        state.node_logs = {}

    state.node_logs["searcher"] = {
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
    },
    "scraping": "enabled",
    "top_k_scraped": 2
    }

    return state