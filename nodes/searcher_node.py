from models.state import ResearchState
from tools.search_tool import search_tool
from services.evaluation.scoring_service import score_results


def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")

    try:
        query = state.query

        results = search_tool(query)

        if not results:
            print("[SEARCHER NODE] No results found")
            state.failed_queries.append(query)
            state.search_retry_count += 1
            return state

        # Apply scoring
        results = score_results(results, query)

        # Store results
        state.search_results[1] = results

        print(f"[SEARCHER NODE] Stored {len(results)} scored results")

        return state

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")
        state.failed_queries.append(state.query)
        return state