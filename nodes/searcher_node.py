import time
from datetime import datetime

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

from utils.logger import log_node_execution
from observability.tracing import trace_node
from config.constants.node_names import NodeNames

from urllib.parse import urlparse


MAX_SEARCH_RETRIES = 3
BACKOFF_BASE = 1 


def _search_with_retry(query, state):
    """
    Retry with exponential backoff
    """
    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            results = search_tool(query)

            if results:
                return results

        except Exception as e:
            state.errors.append(
                ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.search_failure,
                    message=f"Search attempt {attempt+1} failed: {str(e)}"
                )
            )

        # exponential backoff
        time.sleep(BACKOFF_BASE * (2 ** attempt))

    # all retries failed
    state.failure_counts["search_failures"] += 1
    return []


@trace_node(NodeNames.SEARCHER)
def searcher_node(state: ResearchState) -> ResearchState:

    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}
    if not hasattr(state, "failure_counts") or state.failure_counts is None:
        state.failure_counts = {
            "search_failures": 0,
            "parsing_failures": 0,
            "low_confidence": 0,
        }


    state.search_results = {}

    seen_urls = set()
    citation_counter = len(state.citations)

    step_status = {}

    try:
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            query = step.question

            # -------------------------------
            # SEARCH WITH RETRY
            # -------------------------------
            raw_results = _search_with_retry(query, state)

            # fallback query retry
            if not raw_results:
                fallback_query = " ".join(query.split()[:6])
                raw_results = _search_with_retry(fallback_query, state)

            if not raw_results:
                step_status[step_id] = "failed"

                state.errors.append(
                    ErrorLog(
                        node="searcher_node",
                        timestamp=datetime.now().isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"No results after retries for query: {query}"
                    )
                )

                state.search_results[step_id] = []
                continue

            step_status[step_id] = "success"

            # -------------------------------
            # DEDUP
            # -------------------------------
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

            unique_results = unique_results[:5]

            structured_results = []
            MAX_SCRAPES = 3

            for i, (r, norm_url) in enumerate(unique_results):

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title

                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # -------------------------------
                # URL VALIDATION
                # -------------------------------
                try:
                    status = validate_url(norm_url)
                except Exception:
                    status = CitationStatus.broken

                    state.errors.append(
                        ErrorLog(
                            node="searcher_node",
                            timestamp=datetime.now().isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.parsing_error,
                            message=f"URL validation failed: {norm_url}"
                        )
                    )

                # -------------------------------
                # SCRAPING
                # -------------------------------
                content = None
                publish_date = None

                if i < MAX_SCRAPES:
                    try:
                        scraped = scrape_url(norm_url)
                        content = scraped.get("content")
                        publish_date = scraped.get("publish_date")
                    except Exception:
                        state.errors.append(
                            ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.timeout,
                                message=f"Scrape failed: {norm_url}"
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
        state.failure_counts["search_failures"] += 1

        state.errors.append(
            ErrorLog(
                node="searcher_node",
                timestamp=datetime.now().isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=str(e)
            )
        )

    # -------------------------------
    # OBSERVABILITY (IMPORTANT)
    # -------------------------------
    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()}
    )

    node_name = NodeNames.SEARCHER

    existing_log = state.node_logs.get(node_name, {})

    existing_log.update({
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
        },
        "step_status": step_status,
        "total_citations": len(state.citations),
        "search_failures": state.failure_counts["search_failures"],
        "errors_count": len(state.errors)
    })

    state.node_logs[node_name] = existing_log

    state.node_execution_count += 1
    return state