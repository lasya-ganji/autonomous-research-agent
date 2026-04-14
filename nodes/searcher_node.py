import time
from datetime import datetime
from urllib.parse import urlparse

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
from config.constants.node_constants.node_names import NodeNames

from config.constants.scraper_constants import MIN_CONTENT_WORDS
from config.constants.node_constants.search_constants import (
    MAX_SEARCH_RETRIES,
    BACKOFF_BASE,
    MAX_RESULTS_PER_STEP,
    MAX_PER_DOMAIN,
    MAX_SCRAPES
)


# -------------------------------
# SEARCH WITH RETRY 
# -------------------------------
def _search_with_retry(query, state):
    """
    Performs search with limited retry.
    Handles structured tool errors and fail-fast for non-retryable failures.
    """

    for attempt in range(MAX_SEARCH_RETRIES):

        # -------------------------------
        # CALL SEARCH TOOL
        # -------------------------------
        try:
            results = search_tool(query)
        except Exception as e:
            # Exception should be treated as retryable system error
            state.errors.append(
                ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.system_error,
                    message=f"Search attempt {attempt+1} failed for query='{query}': {str(e)}"
                )
            )

            # retry with backoff
            time.sleep(BACKOFF_BASE * (2 ** attempt))
            continue

        # -------------------------------
        # HANDLE STRUCTURED TOOL ERROR (CRITICAL FIRST)
        # -------------------------------
        if isinstance(results, dict) and results.get("error"):

            error_type = results.get("type", "unknown_error")
            severity = results.get("severity", "WARNING")
            retryable = results.get("retryable", True)
            message = results.get("error")

            state.errors.append(
                ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum[severity],
                    error_type=ErrorTypeEnum[error_type],
                    message=f"Search tool error: {message}"
                )
            )

            if not retryable:
                state.api_failure = True
                return []

            # retryable → retry with backoff
            time.sleep(BACKOFF_BASE * (2 ** attempt))
            continue

        # -------------------------------
        # SUCCESS CASE (STRICT)
        # -------------------------------
        if isinstance(results, list) and results:
            return results

        # -------------------------------
        # EMPTY RESULT (NORMAL FAILURE)
        # -------------------------------
        state.errors.append(
            ErrorLog(
                node="searcher_node",
                timestamp=datetime.now().isoformat(),
                severity=SeverityEnum.WARNING,
                error_type=ErrorTypeEnum.search_failure,
                message=f"No results for query: {query} (attempt {attempt+1})"
            )
        )

        # retry only if attempts left
        time.sleep(BACKOFF_BASE * (2 ** attempt))

    # -------------------------------
    # FINAL FAILURE
    # -------------------------------
    state.failure_counts["search_failures"] += 1
    return []

# -------------------------------
# MAIN SEARCHER NODE
# -------------------------------
@trace_node(NodeNames.SEARCHER)
def searcher_node(state: ResearchState) -> ResearchState:

    # -------------------------------
    # STATE INITIALIZATION
    # -------------------------------
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    state.failure_counts["parsing_failures"] = 0
    state.failure_counts["search_failures"] = 0
    state.search_results = {}

    seen_urls = set()
    citation_counter = len(state.citations)
    step_status = {}

    try:
        if not state.research_plan:
            state.errors.append(
                ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="No research plan available"
                )
            )
            return state

        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:

            # -------------------------------
            # HARD STOP ON API FAILURE
            # -------------------------------
            if getattr(state, "api_failure", False):
                break

            step_id = step.step_id
            query = step.question

            print(f"[SEARCH] Step {step_id} Query: {query}")

            # -------------------------------
            # PRIMARY SEARCH
            # -------------------------------
            raw_results = _search_with_retry(query, state)

            if getattr(state, "api_failure", False):
                step_status[step_id] = "failed"
                state.search_results[step_id] = []
                return state

            # -------------------------------
            # FALLBACK SEARCH
            # -------------------------------
            if not raw_results and not getattr(state, "api_failure", False):

                fallback_query = " ".join(query.split()[:6])
                print(f"[SEARCH RETRY] Using fallback query: {fallback_query}")

                raw_results = _search_with_retry(fallback_query, state)

                if getattr(state, "api_failure", False):
                    step_status[step_id] = "failed"
                    state.search_results[step_id] = []
                    return state

            print(f"[SEARCH] Results fetched: {len(raw_results)}")

            # -------------------------------
            # FAILURE CASE
            # -------------------------------
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
                state.unresolved_steps.append(step_id) 
                continue

            step_status[step_id] = "success"

            # -------------------------------
            # DEDUPLICATION
            # -------------------------------
            deduped_results = deduplicate_pipeline(raw_results)

            unique_results = []
            domain_count = {}

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

            unique_results = unique_results[:MAX_RESULTS_PER_STEP]

            structured_results = []

            for i, (r, norm_url) in enumerate(unique_results):

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title

                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # URL validation
                try:
                    status = validate_url(norm_url)
                except Exception as e:
                    state.failure_counts["parsing_failures"] += 1
                    status = CitationStatus.broken

                    state.errors.append(
                        ErrorLog(
                            node="searcher_node",
                            timestamp=datetime.now().isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.system_error,
                            message=f"URL validation failed: {norm_url} | {str(e)}"
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

                        if not content or len(content.split()) < MIN_CONTENT_WORDS:
                            state.failure_counts["parsing_failures"] += 1

                            state.errors.append(
                                ErrorLog(
                                    node="searcher_node",
                                    timestamp=datetime.now().isoformat(),
                                    severity=SeverityEnum.WARNING,
                                    error_type=ErrorTypeEnum.parsing_error,
                                    message=f"No usable content: {norm_url}"
                                )
                            )

                    except Exception as e:
                        state.failure_counts["parsing_failures"] += 1

                        state.errors.append(
                            ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.system_error,
                                message=f"Scrape failed: {norm_url} | {str(e)}"
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
    # OBSERVABILITY
    # -------------------------------
    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()}
    )

    existing_log = state.node_logs.get(NodeNames.SEARCHER, {})

    existing_log.update({
        "total_steps": len(state.research_plan),
        "results_per_step": {k: len(v) for k, v in state.search_results.items()},
        "step_status": step_status,
        "successful_steps": sum(1 for s in step_status.values() if s == "success"),
        "failed_steps": sum(1 for s in step_status.values() if s == "failed"),
        "total_sources": len(state.citations),
        "search_failures": state.failure_counts["search_failures"],
        "parsing_failures": state.failure_counts["parsing_failures"],
        "errors_count": len(state.errors)
    })

    state.node_logs[NodeNames.SEARCHER] = existing_log

    state.node_execution_count += 1
    return state