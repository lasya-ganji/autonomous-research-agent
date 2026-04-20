import re
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
from tools.scraper_tool import scrape_url, is_usable_source

from services.retrieval.dedup_service import (
    deduplicate_pipeline,
    normalize_url
)
from services.citation.citation_service import validate_url

from utils.logger import log_node_execution
from observability.tracing import trace_node
from config.constants.node_constants.node_names import NodeNames

from config.constants.scraper_constants import (
    MIN_CONTENT_WORDS, SCRAPE_MIN_RELEVANCE, DOMAIN_FAIL_THRESHOLD,
    NON_HTML_EXTENSIONS, MIN_SNIPPET_WORDS, UNSEARCHABLE_URL_PATTERNS,
)
from config.constants.node_constants.search_constants import (
    MAX_SEARCH_RETRIES,
    BACKOFF_BASE,
    MAX_RESULTS_PER_STEP,
    MAX_PER_DOMAIN,
    MAX_SCRAPES,
    STOPWORDS,
)


def _pre_scrape_score(result, query: str) -> float:
    """
    Lightweight composite score computed from already-available metadata.
    Used to rank candidates before scraping so scrape slots go to the best sources.

    Signals:
      - Tavily relevance score (0.5 weight): production ML signal, already on result
      - Snippet word density  (0.3 weight): longer snippets predict richer page content
      - Title-query Jaccard   (0.2 weight): topical alignment without domain lists
    """
    tavily_score = float(getattr(result, "relevance_score", 0.5) or 0.5)

    snippet = (getattr(result, "snippet", "") or "").strip()
    snippet_density = min(len(snippet.split()) / 60.0, 1.0)

    query_terms = set(query.lower().split()) - STOPWORDS
    title_terms = set((getattr(result, "title", "") or "").lower().split()) - STOPWORDS
    union = query_terms | title_terms
    title_overlap = len(query_terms & title_terms) / len(union) if union else 0.0

    score = 0.5 * tavily_score + 0.3 * snippet_density + 0.2 * title_overlap
    return round(min(max(score, 0.0), 1.0), 3)


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
        # HANDLE STRUCTURED TOOL ERROR 
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
                    severity=SeverityEnum[severity] if severity in SeverityEnum.__members__ else SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum(error_type),
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
        # SUCCESS CASE 
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

    state.search_results = {}

    seen_urls = set()
    citation_counter = len(state.citations)
    step_status = {}
    filtered_count = 0
    snippet_fallback_count = 0
    failed_domains: dict[str, int] = {}  # runtime domain failure tracker — resets each node call

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
            step_filtered = 0

            for r in deduped_results:
                url = str(getattr(r, "url", ""))
                if not url:
                    continue

                norm_url = normalize_url(url)
                snippet_text = (getattr(r, "snippet", "") or "").strip()
                snippet_words = len(snippet_text.split())
                snippet_preview = snippet_text[:80] + ("..." if len(snippet_text) > 80 else "")
                parsed_path = urlparse(norm_url).path.lower()  # used by Gate 1

                print(f"[CANDIDATE] url={norm_url} snippet_words={snippet_words} snippet='{snippet_preview}'")

                # -------------------------------
                # URL FILTER (pre-scrape)
                # -------------------------------
                # Gate 1: binary/non-HTML file extension
                if any(parsed_path.endswith(ext) for ext in NON_HTML_EXTENSIONS):
                    print(f"[FILTER:gate1:extension] DROPPED url={norm_url} reason=non-html-extension path={parsed_path}")
                    step_filtered += 1
                    filtered_count += 1
                    continue

                # Gate 2: URL structural pattern — content-type classification without domain names.
                # Identifies VIDEO pages (/watch?v=), SHORT VIDEO (/shorts/), REELS (/reel/),
                # SOCIAL POSTS (/status/12345), FORUM THREADS (/r/name/comments/).
                # Generalises: any future platform using the same URL conventions is caught.
                parsed = urlparse(norm_url)
                path_and_query = parsed.path + ("?" + parsed.query if parsed.query else "")
                matched_pattern = next(
                    (p for p in UNSEARCHABLE_URL_PATTERNS if re.search(p, path_and_query, re.IGNORECASE)),
                    None
                )
                if matched_pattern:
                    print(f"[FILTER:gate2:url_pattern] DROPPED url={norm_url} matched_pattern='{matched_pattern}'")
                    step_filtered += 1
                    filtered_count += 1
                    continue

                # Gate 3: snippet richness — catches auth/login-wall pages not caught by URL patterns
                if snippet_words < MIN_SNIPPET_WORDS:
                    print(f"[FILTER:gate3:snippet] DROPPED url={norm_url} reason=snippet_too_short snippet_words={snippet_words} snippet='{snippet_preview}'")
                    step_filtered += 1
                    filtered_count += 1
                    continue

                domain = urlparse(norm_url).netloc

                # Gate 4: runtime domain failure tracker — skip domains that
                # have already failed repeatedly in this run (auth_blocked / timeout / network)
                domain_failures = failed_domains.get(domain, 0)
                if domain_failures >= DOMAIN_FAIL_THRESHOLD:
                    print(f"[FILTER:gate4:runtime] DROPPED url={norm_url} domain={domain} accumulated_failures={domain_failures}")
                    step_filtered += 1
                    filtered_count += 1
                    continue

                if domain_count.get(domain, 0) >= MAX_PER_DOMAIN:
                    print(f"[FILTER:domain_cap] DROPPED url={norm_url} domain={domain} already_have={domain_count[domain]}/{MAX_PER_DOMAIN}")
                    continue

                if norm_url in seen_urls:
                    print(f"[FILTER:dedup] DROPPED url={norm_url} reason=already_seen_across_steps")
                    continue

                seen_urls.add(norm_url)
                domain_count[domain] = domain_count.get(domain, 0) + 1
                print(f"[FILTER:PASS] ACCEPTED url={norm_url} domain={domain} snippet_words={snippet_words}")

                unique_results.append((r, norm_url))

            # Sort by pre-scrape score so scrape slots go to the best candidates.
            # Sources below MAX_SCRAPES rank still enter as snippet-only — nothing is dropped.
            unique_results.sort(key=lambda x: _pre_scrape_score(x[0], query), reverse=True)
            unique_results = unique_results[:MAX_RESULTS_PER_STEP]

            print(f"[STEP {step_id}] Candidates after filtering: {len(unique_results)}")

            structured_results = []
            step_snippet_fallbacks = 0

            for i, (r, norm_url) in enumerate(unique_results):
                initial_score = _pre_scrape_score(r, query)
                domain = urlparse(norm_url).netloc

                print(f"\n[SCRAPE LOOP] rank={i+1}/{len(unique_results)} pre_scrape={initial_score:.3f} url={norm_url}")

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title

                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # -------------------------------
                # URL VALIDATION
                # -------------------------------
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

                # -------------------------------
                # SCRAPING + FALLBACK
                # -------------------------------
                # Prefer Tavily server content; if unavailable and relevance is high, scrape locally. Otherwise, use snippet only.
         
                tavily_content = getattr(r, "content", None)
                content = tavily_content if tavily_content else None
                publish_date = None

                if content:
                    print(f"[CONTENT:tavily] url={norm_url} words={len(content.split())} → using Tavily pre-fetched content")

                elif i < MAX_SCRAPES and initial_score >= SCRAPE_MIN_RELEVANCE:
                    print(f"[CONTENT:scrape] url={norm_url} rank={i+1} score={initial_score:.3f} → attempting local scrape")
                    try:
                        scraped = scrape_url(norm_url)
                        scrape_status = scraped.get("status", "failed")

                        if scrape_status == "success":
                            content = scraped.get("content")
                            publish_date = scraped.get("publish_date")
                            print(f"[CONTENT:scrape:SUCCESS] url={norm_url} words={len((content or '').split())}")

                        elif scrape_status == "low_content":
                            print(f"[CONTENT:scrape:LOW_CONTENT] url={norm_url} → snippet fallback (page exists but text too thin)")
                            state.errors.append(
                                ErrorLog(
                                    node="searcher_node",
                                    timestamp=datetime.now().isoformat(),
                                    severity=SeverityEnum.WARNING,
                                    error_type=ErrorTypeEnum.parsing_error,
                                    message=f"[QUALITY] Rejected low content url={norm_url} → snippet fallback"
                                )
                            )
                            step_snippet_fallbacks += 1
                            snippet_fallback_count += 1

                        else:
                            # scrape_status == "failed"
                            state.failure_counts["parsing_failures"] += 1
                            error_type = scraped.get("error_type")
                            print(f"[CONTENT:scrape:FAILED] url={norm_url} error_type={error_type} → snippet fallback")
                            state.errors.append(
                                ErrorLog(
                                    node="searcher_node",
                                    timestamp=datetime.now().isoformat(),
                                    severity=SeverityEnum.WARNING,
                                    error_type=ErrorTypeEnum.system_error,
                                    message=f"[SCRAPER ERROR] url={norm_url} error_type={error_type} → snippet fallback"
                                )
                            )
                            # Increment domain failure counter for auth/network failures.
                            # low_content is a page quality issue, not a domain-level block,
                            # so it is intentionally excluded from runtime tracking.
                            if error_type in {"auth_blocked", "timeout_error", "network_error"}:
                                failed_domains[domain] = failed_domains.get(domain, 0) + 1
                                print(f"[RUNTIME:domain_fail] domain={domain} total_failures={failed_domains[domain]} (threshold={DOMAIN_FAIL_THRESHOLD})")
                            step_snippet_fallbacks += 1
                            snippet_fallback_count += 1

                    except Exception as e:
                        state.failure_counts["parsing_failures"] += 1
                        print(f"[CONTENT:scrape:EXCEPTION] url={norm_url} error={e} → snippet fallback")
                        state.errors.append(
                            ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.system_error,
                                message=f"[SCRAPER ERROR] url={norm_url} reason={str(e)} → snippet fallback"
                            )
                        )
                        step_snippet_fallbacks += 1
                        snippet_fallback_count += 1

                else:
                    if i >= MAX_SCRAPES:
                        reason = f"rank_limit (rank={i+1} > MAX_SCRAPES={MAX_SCRAPES})"
                    else:
                        reason = f"low_relevance (score={initial_score:.3f} < SCRAPE_MIN_RELEVANCE={SCRAPE_MIN_RELEVANCE})"
                    print(f"[CONTENT:snippet_only] url={norm_url} reason={reason}")

                # content=None → synthesiser falls back to snippet automatically
                result = SearchResult(
                    citation_id=citation_id,
                    url=norm_url,
                    title=title,
                    snippet=snippet,
                    content=content,
                    publish_date=publish_date,
                    quality_score=initial_score,   
                    relevance_score=initial_score,
                    recency_score=0.5,
                    domain_score=0.5,
                    depth_score=0.5,
                    rank=i + 1
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

            # If all results were filtered out, mark step_status as "no_content" for accurate evaluation.
     
            if not structured_results and step_status.get(step_id) == "success":
                step_status[step_id] = "no_content"
         
                state.unresolved_steps.append(step_id)
                state.errors.append(ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.search_failure,
                    message=f"Step {step_id}: all results filtered or already seen"
                ))

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
        "urls_filtered": filtered_count,
        "snippet_fallbacks": snippet_fallback_count,
        "errors_count": len(state.errors)
    })

    state.node_logs[NodeNames.SEARCHER] = existing_log

    return state