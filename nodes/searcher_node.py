from models.state import ResearchState 
from models.search_models import SearchResult
from models.citation_models import Citation
from models.enums import CitationStatus

from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url

from services.retrieval.dedup_service import (
    deduplicate_pipeline,
    normalize_url
)
from services.citation.citation_service import validate_url
from services.retrieval.embedding_service import get_embedding

from utils.logger import log_node_execution
from observability.tracing import trace_node
from config.constants.node_names import NodeNames

from urllib.parse import urlparse

from datetime import datetime


@trace_node(NodeNames.SEARCHER)
def searcher_node(state: ResearchState) -> ResearchState:
    
    if state.errors is None:
        state.errors = []
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

            # SEARCH
            raw_results = search_tool(query)

            if not raw_results:
                fallback_query = " ".join(query.split()[:6])
                raw_results = search_tool(fallback_query)

            raw_results = raw_results or []

            
            if not raw_results:
                state.errors.append(
                    ErrorLog(
                        node="searcher_node",
                        timestamp=datetime.now().isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"No results found for query: {query}"
                    )
                )

            # DEDUP PIPELINE
            deduped_results = deduplicate_pipeline(raw_results)
            
            
            unique_results = []

            domain_count = {}
            MAX_PER_DOMAIN = 2

            for r in deduped_results:
                url = str(getattr(r, "url", ""))
                if not url:
                    continue

                norm_url = normalize_url(url)
                domain = urlparse(norm_url).netloc  

                if domain_count.get(domain, 0) >= MAX_PER_DOMAIN:
                    continue

                if norm_url in seen_urls:
                    continue

                seen_urls.add(norm_url)
                domain_count[domain] = domain_count.get(domain, 0) + 1

                unique_results.append((r, norm_url))

            # LIMIT RESULTS
            MAX_RESULTS_PER_STEP = 5
            unique_results = unique_results[:MAX_RESULTS_PER_STEP]

            # STRUCTURE RESULTS
            structured_results = []

            MAX_SCRAPES = 3

            for i, (r, norm_url) in enumerate(unique_results):

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title

                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # Validate URL
                try:
                    status = validate_url(norm_url)
                except Exception:
                    status = CitationStatus.broken

                    # ADDED
                    state.errors.append(
                        ErrorLog(
                            node="searcher_node",
                            timestamp=datetime.now().isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.search_failure,
                            message=f"URL validation failed: {norm_url}"
                        )
                    )

                # Scraping 
                content = None
                publish_date = None

                if i < MAX_SCRAPES:
                    try:
                        scraped = scrape_url(norm_url)
                        content = scraped.get("content")
                        publish_date = scraped.get("publish_date")
                    except Exception:
                        # ADDED
                        state.errors.append(
                            ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.timeout,
                                message=f"Scrape failed for URL: {norm_url}"
                            )
                        )

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

        # critical error logging
        state.errors.append(
            ErrorLog(
                node="searcher_node",
                timestamp=datetime.now().isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.search_failure,
                message=str(e)
            )
        )

    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()}
    )

    existing_log = state.node_logs.get(NodeNames.SEARCHER, {})

    existing_log.update({
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
        },
        "total_citations": len(state.citations),
        "errors_count": len(state.errors)
    })

    state.node_logs[NodeNames.SEARCHER] = existing_log

    state.node_execution_count += 1

    return state