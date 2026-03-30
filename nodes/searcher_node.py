from models.state import ResearchState
from models.search_models import SearchResult

from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url

from utils.logger import log_node_execution
from observability.tracing import trace_node

import time
import uuid


@trace_node("searcher_node")
def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")

    start_time = time.time()

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    state.search_results = {}

    try:
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            raw_results = search_tool(query)

            structured_results = []
            seen_urls = set()

            # scraping limits per priority
            scrape_limits = {
                1: 3,
                2: 2,
                3: 1
            }

            max_scrapes = scrape_limits.get(step.priority, 1)

            if raw_results:
                for i, r in enumerate(raw_results):

                    url = getattr(r, "url", None)
                    title = getattr(r, "title", "Untitled")
                    snippet = getattr(r, "snippet", "")

                    if not url or url in seen_urls:
                        continue

                    seen_urls.add(url)

                    # Generate unique citation_id
                    citation_id = str(uuid.uuid4())

                    # Safe scraping
                    content = None
                    if i < max_scrapes:
                        try:
                            content = scrape_url(url)
                        except Exception as e:
                            print(f"[SCRAPER ERROR] {url}: {e}")

                    # fallback if snippet empty
                    if not snippet:
                        snippet = title

                    result = SearchResult(
                        citation_id=citation_id,
                        url=url,
                        title=title,
                        snippet=snippet,
                        content=content,

                        # placeholders (will be overwritten)
                        quality_score=0.5,
                        relevance_score=0.5,
                        recency_score=0.5,
                        domain_score=0.5,
                        depth_score=0.5,
                        rank=1
                    )

                    structured_results.append(result)

            state.search_results[step_id] = structured_results

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")

    # Logger
    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()},
        start_time=start_time
    )

    # Structured logs
    if state.node_logs is None:
        state.node_logs = {}

    state.node_logs["searcher"] = {
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
        },
        "scraping": "enabled",
        "top_k_scraped": 3
    }

    state.node_execution_count += 1

    return state