from models.state import ResearchState
from models.search_models import SearchResult
from models.citation_models import Citation
from models.enums import CitationStatus

from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url

from services.retrieval.dedup_service import deduplicate_results
from services.citation.citation_service import validate_url

from utils.logger import log_node_execution
from observability.tracing import trace_node

import time
from datetime import datetime


@trace_node("searcher_node")
def searcher_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    state.search_results = {}

    seen_urls = set()
    citation_counter = len(state.citations)

    try:
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            query = step.question

            raw_results = search_tool(query)

            if not raw_results:
                fallback_query = " ".join(query.split()[:6])
                raw_results = search_tool(fallback_query)

            unique_results = deduplicate_results(raw_results or [], seen_urls)

            structured_results = []

            scrape_limits = {1: 3, 2: 2, 3: 1}
            max_scrapes = scrape_limits.get(step.priority, 1)
            scrape_count = 0

            for r, norm_url in unique_results:

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title

                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # validate EARLY
                try:
                    status = validate_url(norm_url)
                except Exception:
                    status = CitationStatus.broken

                content = None
                if scrape_count < max_scrapes:
                    try:
                        scraped = scrape_url(norm_url)

                        content = scraped.get("content")
                        publish_date = scraped.get("publish_date")
                        
                        scrape_count += 1
                    except Exception:
                        pass

                result = SearchResult(
                    citation_id=citation_id,
                    url=norm_url,
                    title=title,
                    snippet=snippet,
                    content=content,
                    publish_date=publish_date,
                    quality_score=0.5,
                    relevance_score=0.5,
                    recency_score=0.5,
                    domain_score=0.5,
                    depth_score=0.5,
                    rank=1
                )

                structured_results.append(result)

                # source of truth
                state.citations[citation_id] = Citation(
                    citation_id=citation_id,
                    title=title,
                    url=norm_url,
                    quality_score=0.5,
                    status=status,
                    date_accessed=datetime.now().isoformat()
                )

            state.search_results[step_id] = structured_results

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")

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
        "total_citations": len(state.citations)
    }

    state.node_execution_count += 1

    return state