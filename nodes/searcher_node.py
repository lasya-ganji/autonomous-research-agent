from models.state import ResearchState
from tools.search_tool import search_tool
from services.evaluation.scoring_service import score_results


def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")

    try:
        # Loop through each planned step
        for step in state.research_plan:
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            results = search_tool(query)

            # Handle no results
            if not results:
                print(f"[SEARCHER NODE] No results for step {step_id}")
                state.failed_queries.append(query)
                continue

            # Apply scoring
            results = score_results(results, query)

            # Store results per step_id
            state.search_results[step_id] = results

            print(f"[SEARCHER NODE] Stored {len(results)} results for step {step_id}")

        # Update unresolved steps
        state.unresolved_steps = [
            step.step_id for step in state.research_plan
            if step.step_id not in state.search_results
        ]

        # Retry logic
        if not state.search_results:
            state.search_retry_count += 1

        return state

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")
        state.failed_queries.append(state.query)
        return state