from models.state import ResearchState
from models.search_models import SearchResult
from models.citation_models import Citation
from models.enums import CitationStatus

from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url

from utils.logger import log_node_execution
from observability.tracing import trace_node

from urllib.parse import urlparse

import time
from datetime import datetime


def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except:
        return url


@trace_node("searcher_node")
def searcher_node(state: ResearchState) -> ResearchState:
    print("Searcher Node")

    start_time = time.time()

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    # DO NOT reset citations
    if state.citations is None:
        state.citations = {}

    state.search_results = {}

    # safer counter (monotonic)
    citation_counter = len(state.citations)

    try:
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            query = step.question

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            raw_results = search_tool(query)

            structured_results = []

            scrape_limits = {1: 3, 2: 2, 3: 1}
            max_scrapes = scrape_limits.get(step.priority, 1)
            scrape_count = 0

            if raw_results:
                for r in raw_results:

                    url = getattr(r, "url", None)
                    title = getattr(r, "title", "Untitled")
                    snippet = getattr(r, "snippet", "")

                    if not url:
                        continue

                    norm_url = normalize_url(url)

                    # GLOBAL dedup (across all steps)
                    if norm_url in state.deduplicated_urls:
                        continue

                    state.deduplicated_urls.add(norm_url)

                    # stable incremental ID
                    citation_counter += 1
                    citation_id = f"[{citation_counter}]"

                    # safe scraping
                    content = None
                    if scrape_count < max_scrapes:
                        try:
                            content = scrape_url(norm_url)
                            scrape_count += 1
                        except Exception as e:
                            print(f"[SCRAPER ERROR] {norm_url}: {e}")

                    if not snippet:
                        snippet = title

                    result = SearchResult(
                        citation_id=citation_id,
                        url=norm_url,
                        title=title,
                        snippet=snippet,
                        content=content,

                        quality_score=0.5,
                        relevance_score=0.5,
                        recency_score=0.5,
                        domain_score=0.5,
                        depth_score=0.5,
                        rank=1
                    )

                    structured_results.append(result)

                    # store citation safely
                    if citation_id not in state.citations:
                        state.citations[citation_id] = Citation(
                            citation_id=citation_id,
                            title=title,
                            url=norm_url,
                            quality_score=0.5,
                            status=CitationStatus.valid,
                            date_accessed=datetime.now().isoformat()
                        )

            state.search_results[step_id] = structured_results

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")

    # LOGGING

    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()},
        start_time=start_time
    )

    state.node_logs["searcher"] = {
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
        },
        "scraping": "enabled",
        "top_k_scraped": 3,
        "total_citations": len(state.citations)
    }

    state.node_execution_count += 1

    return state