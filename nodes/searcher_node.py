import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from models.state import ResearchState
from models.search_models import SearchResult
from models.citation_models import Citation
from models.enums import CitationStatus

from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from tools.search_tool import search_tool
from tools.scraper_tool import scrape_url
from tools.llm_tool import call_llm

from services.retrieval.dedup_service import (
    deduplicate_pipeline,
    normalize_url
)
from services.citation.citation_service import validate_url, status_from_scrape

from utils.logger import log_node_execution
from observability.tracing import trace_node
from config.constants.node_constants.node_names import NodeNames

from config.constants.scraper_constants import (
    MIN_CONTENT_WORDS, SCRAPE_MIN_RELEVANCE, DOMAIN_FAIL_THRESHOLD,
    MIN_SNIPPET_WORDS, BINARY_EXTENSIONS, NON_ARTICLE_URL_PATTERNS,
    UGC_DROP_URL_PATTERNS, CURATE_AUTO_ACCEPT_SCORE,
)
from config.constants.node_constants.search_constants import (
    MAX_SEARCH_RETRIES,
    BACKOFF_BASE,
    MAX_RESULTS_PER_STEP,
    MAX_PER_DOMAIN,
    MAX_SCRAPES,
    STOPWORDS,
)


# -----------------------------------------------
# PRE-SCRAPE RANKING
# -----------------------------------------------
def _pre_scrape_score(result, query: str) -> float:
    """
    Lightweight composite score from already-available metadata.
    Ranks candidates before scraping so scrape slots go to the best sources.

    Signals:
      - Tavily relevance score (0.5 weight)
      - Snippet word density  (0.3 weight)
      - Title-query Jaccard   (0.2 weight)
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


# -----------------------------------------------
# LLM SOURCE CURATION — THREE-TIER FALLBACK
# -----------------------------------------------

# LLM page_type values that represent UGC/social/forum content.
# Used as a post-LLM veto: even if the LLM says "accept", these types are
# rejected because they are not citable in a research document.
# Catches LinkedIn posts, Facebook groups, etc. that slip past URL-pattern gates.
_UGC_PAGE_TYPES = frozenset({
    "forum_thread", "social_post", "user_profile", "qa_listing",
})

# Observable snippet text patterns that definitively indicate non-research content.
# These specific phrases do not appear in research articles — only in login walls,
# paywalls, marketing pages, and navigation-only pages.
# Domain-agnostic: detects content type from language, not from site names.
_FALLBACK_REJECT_SIGNALS = re.compile(
    r'sign in to continue|please log in|subscribe to access|'
    r'create an account to|members only|log in to view|'
    r'sign up for free|get started (?:today|now)|'
    r'buy now|skip to (?:main )?content',
    re.IGNORECASE,
)


def _signal_fallback_filter(candidates: list) -> list:
    """
    Tier-2 fallback when LLM curation times out or fails.

    Two-pass approach:

    Pass A — snippet text signals:
        Rejects snippets with observable non-research language patterns
        (login/paywall prompts, marketing CTAs, navigation-only text).
        Domain-agnostic — uses content language, not site names.

    Pass B — numeric baseline:
        Keeps candidates with snippet_words >= MIN_SNIPPET_WORDS
        and relevance_score >= SCRAPE_MIN_RELEVANCE.

    Tier hierarchy:
      Tier 2  — intersection of Pass A and Pass B (text-clean AND numerically qualified)
      Tier 2b — Pass B only, if Pass A+B combined is empty (numeric safety net)
      Tier 3  — all candidates, if even Pass B is empty (pipeline starvation prevention)
    """
    # Pass A: reject snippets matching non-research language signals
    pass_a_urls = {
        url for r, url in candidates
        if not _FALLBACK_REJECT_SIGNALS.search(getattr(r, "snippet", "") or "")
    }

    # Pass B: numeric quality baseline
    pass_b = [
        (r, url) for r, url in candidates
        if len((getattr(r, "snippet", "") or "").split()) >= MIN_SNIPPET_WORDS
        and float(getattr(r, "relevance_score", 0.0) or 0.0) >= SCRAPE_MIN_RELEVANCE
    ]

    # Tier 2: candidates passing both passes
    combined = [(r, url) for r, url in pass_b if url in pass_a_urls]
    if combined:
        print(f"[CURATION:FALLBACK] text+numeric filter: {len(combined)}/{len(candidates)} passed")
        return combined

    # Tier 2b: text signals cleared everything — fall back to numeric-only
    if pass_b:
        print(f"[CURATION:FALLBACK] text signals empty — numeric-only: {len(pass_b)}/{len(candidates)} passed")
        return pass_b

    # Tier 3: both passes empty — pass all to prevent pipeline starvation
    print(f"[CURATION:EMERGENCY] all fallback filters empty — passing all {len(candidates)} candidates")
    return candidates


def _load_curator_prompt() -> str | None:
    """
    Loads the curator prompt from disk. Returns None on any I/O failure,
    signalling the caller to skip LLM curation entirely.
    """
    try:
        prompt_path = Path(__file__).parent.parent / "prompts" / "source_curator.txt"
        return prompt_path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"[CURATION:PROMPT_MISSING] cannot load source_curator.txt ({e})")
        return None


def _parse_llm_decisions(raw_content: str) -> dict | None:
    """
    Parses LLM curation JSON into url → {decision, page_type} mapping.
    Returns None on any parse failure (caller should fall back).
    """
    if not raw_content.strip():
        print("[CURATION:LLM_EMPTY] empty response — falling back")
        return None
    try:
        decisions = json.loads(raw_content)
    except json.JSONDecodeError as e:
        preview = raw_content[:120].replace("\n", " ")
        print(f"[CURATION:PARSE_ERROR] invalid JSON ({e}) preview='{preview}' — falling back")
        return None
    if not isinstance(decisions, list):
        print(f"[CURATION:SHAPE_ERROR] expected array, got {type(decisions).__name__} — falling back")
        return None
    return {
        d["url"]: {
            "decision": d.get("decision", "accept"),
            "page_type": d.get("page_type", "other"),
        }
        for d in decisions
        if isinstance(d, dict) and d.get("url")
    }


def _curate_sources(candidates: list, query: str) -> list:
    """
    Two-tier source curation — semantic filter before any scraping.

    Tier 0 — auto-accept (no LLM cost):
      Sources with sufficient Tavily pre-fetched content (>= MIN_CONTENT_WORDS)
      AND a confident relevance score (>= CURATE_AUTO_ACCEPT_SCORE) are accepted
      immediately. UGC/forum URLs are removed before this stage by Phase-1
      Gate 2, so Tier 0 no longer needs a structural guard here.

    Tier 1 — LLM semantic judgment (30s timeout):
      Ambiguous candidates (no Tavily body, low score, short snippet) are
      evaluated for relevance, substance, and citability. The LLM also
      identifies residual UGC/social content (e.g. LinkedIn posts, Facebook
      groups) that slipped past URL patterns; a post-LLM veto enforces
      rejection for those page_types even if the model said "accept".
      Falls back to text+numeric signal filter when LLM fails.
    """
    if not candidates:
        return candidates

    # ── Tier 0: auto-accept Tavily-validated sources ──
    auto_accepted = []
    needs_llm = []
    for r, url in candidates:
        content_words = len((getattr(r, "content", None) or "").split())
        score = float(getattr(r, "relevance_score", 0.0) or 0.0)
        if content_words >= MIN_CONTENT_WORDS and score >= CURATE_AUTO_ACCEPT_SCORE:
            auto_accepted.append((r, url))
        else:
            needs_llm.append((r, url))

    print(f"[CURATION] tier0_auto={len(auto_accepted)} tier1_llm={len(needs_llm)}")

    if not needs_llm:
        print(f"[CURATION:SKIP] all {len(auto_accepted)} candidates auto-accepted — LLM skipped")
        return auto_accepted

    # ── Tier 1: LLM for ambiguous candidates ──
    template = _load_curator_prompt()
    if template is None:
        return auto_accepted + _signal_fallback_filter(needs_llm)

    sources_payload = json.dumps([
        {
            "url": url,
            "title": getattr(r, "title", "") or "",
            "snippet": (getattr(r, "snippet", "") or "")[:300],
        }
        for r, url in needs_llm
    ], indent=2)
    prompt = template.replace("{{QUERY}}", query).replace("{{SOURCES}}", sources_payload)

    response = call_llm(prompt, temperature=0.0, timeout=30.0)

    if response.get("error"):
        err_type = response.get("error_type", "unknown_error")
        print(f"[CURATION:LLM_ERROR] type={err_type} msg={response.get('error')} — falling back")
        return auto_accepted + _signal_fallback_filter(needs_llm)

    url_to_decision = _parse_llm_decisions(response.get("content") or "")
    if url_to_decision is None:
        return auto_accepted + _signal_fallback_filter(needs_llm)

    # Post-LLM veto: reject UGC page_types even when the model says "accept".
    # Catches social/forum content the LLM labelled as accepted but cannot cite.
    llm_accepted = []
    for r, url in needs_llm:
        entry = url_to_decision.get(url, {"decision": "accept", "page_type": "other"})
        page_type = entry.get("page_type", "other")
        if entry.get("decision") == "accept":
            if page_type in _UGC_PAGE_TYPES:
                print(f"[CURATION:LLM_VETO] url={url} page_type={page_type} → UGC rejected")
            else:
                llm_accepted.append((r, url))

    print(f"[CURATION:LLM] {len(llm_accepted)}/{len(needs_llm)} ambiguous accepted")
    return auto_accepted + llm_accepted


# -----------------------------------------------
# SEARCH WITH RETRY
# -----------------------------------------------
def _search_with_retry(query: str, state: ResearchState, exclude_domains: list = None):
    """
    Performs a Tavily search with retry + exponential backoff.
    Passes dynamically learned domain exclusions to the API.
    """
    for attempt in range(MAX_SEARCH_RETRIES):

        try:
            results = search_tool(query, exclude_domains=exclude_domains)
        except Exception as e:
            state.errors.append(
                ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.system_error,
                    message=f"Search attempt {attempt+1} failed for query='{query}': {str(e)}"
                )
            )
            time.sleep(BACKOFF_BASE * (2 ** attempt))
            continue

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

            time.sleep(BACKOFF_BASE * (2 ** attempt))
            continue

        if isinstance(results, list) and results:
            return results

        state.errors.append(
            ErrorLog(
                node="searcher_node",
                timestamp=datetime.now().isoformat(),
                severity=SeverityEnum.WARNING,
                error_type=ErrorTypeEnum.search_failure,
                message=f"No results for query: {query} (attempt {attempt+1})"
            )
        )
        time.sleep(BACKOFF_BASE * (2 ** attempt))

    state.failure_counts["search_failures"] += 1
    return []


# -----------------------------------------------
# MAIN SEARCHER NODE
# -----------------------------------------------
@trace_node(NodeNames.SEARCHER)
def searcher_node(state: ResearchState) -> ResearchState:

    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    state.search_results = {}

    seen_urls: set = set()
    citation_counter = len(state.citations)
    step_status: dict = {}
    filtered_count = 0
    snippet_fallback_count = 0
    per_step_metrics: dict = {}
    # Tracks per-domain scrape failures across the entire run AND across supervisor
    # re-search loops. Domains reaching DOMAIN_FAIL_THRESHOLD are excluded from
    # subsequent Tavily searches. Pulling from state so learning is not lost when
    # the supervisor re-invokes the searcher.
    failed_domains: dict[str, int] = dict(getattr(state, "failed_domains", {}) or {})

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

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1 — Search + structural filter + dedup + snippet gates
        # Collects filtered candidates for every step WITHOUT calling the
        # LLM yet, so Phase 2 can batch all steps into a single LLM call.
        # ═══════════════════════════════════════════════════════════════
        # Per-step state accumulated during Phase 1, consumed in Phase 3.
        phase1: dict = {}  # step_id → {query, candidates, raw_len, pre_filtered_len, deduped_len, step_filtered}

        for step in steps:
            if getattr(state, "api_failure", False):
                break

            step_id = step.step_id
            query = step.question

            print(f"\n[SEARCH] Step {step_id} | Query: {query}")

            learned_exclusions = [
                domain for domain, count in failed_domains.items()
                if count >= DOMAIN_FAIL_THRESHOLD
            ]
            if learned_exclusions:
                print(f"[SEARCH] Excluding {len(learned_exclusions)} failed domains from this step")

            # ── Primary search ──
            raw_results = _search_with_retry(query, state, exclude_domains=learned_exclusions)

            if getattr(state, "api_failure", False):
                step_status[step_id] = "failed"
                state.search_results[step_id] = []
                return state

            # ── Fallback search (shortened query) ──
            if not raw_results and not getattr(state, "api_failure", False):
                fallback_query = " ".join(query.split()[:6])
                print(f"[SEARCH RETRY] Fallback query: {fallback_query}")
                raw_results = _search_with_retry(fallback_query, state, exclude_domains=learned_exclusions)
                if getattr(state, "api_failure", False):
                    step_status[step_id] = "failed"
                    state.search_results[step_id] = []
                    return state

            if not raw_results:
                step_status[step_id] = "failed"
                state.errors.append(ErrorLog(
                    node="searcher_node",
                    timestamp=datetime.now().isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.search_failure,
                    message=f"No results after retries for query: {query}"
                ))
                state.search_results[step_id] = []
                state.unresolved_steps.append(step_id)
                phase1[step_id] = {"query": query, "candidates": [], "raw_len": 0,
                                   "pre_filtered_len": 0, "deduped_len": 0, "step_filtered": 0}
                continue

            step_status[step_id] = "success"

            # ── Stage A: Structural pre-filter ──
            # Binary files, video player paths, social post IDs → dropped before
            # the embedding-based dedup to avoid wasting embedding API calls.
            step_filtered = 0
            pre_filtered = []
            for r in raw_results:
                url = str(getattr(r, "url", ""))
                if not url:
                    continue
                norm_url = normalize_url(url)
                if BINARY_EXTENSIONS.search(norm_url):
                    print(f"[FILTER:gate1:extension] DROPPED url={norm_url} reason=binary_file_extension")
                    step_filtered += 1
                    filtered_count += 1
                    continue
                if NON_ARTICLE_URL_PATTERNS.search(norm_url):
                    print(f"[FILTER:gate2:url_pattern] DROPPED url={norm_url} reason=non_article_url_pattern")
                    step_filtered += 1
                    filtered_count += 1
                    continue
                if UGC_DROP_URL_PATTERNS.search(norm_url):
                    print(f"[FILTER:gate2:ugc_pattern] DROPPED url={norm_url} reason=ugc_url_pattern")
                    step_filtered += 1
                    filtered_count += 1
                    continue
                pre_filtered.append(r)

            # ── Stage B: Deduplication (URL → heuristic → semantic) ──
            deduped_results = deduplicate_pipeline(pre_filtered)

            # ── Stage C: Snippet quality + domain caps + cross-step dedup ──
            unique_results = []
            domain_count: dict = {}

            for r in deduped_results:
                url = str(getattr(r, "url", ""))
                if not url:
                    continue
                norm_url = normalize_url(url)
                snippet_text = (getattr(r, "snippet", "") or "").strip()
                snippet_words = len(snippet_text.split())
                snippet_preview = snippet_text[:80] + ("..." if len(snippet_text) > 80 else "")

                print(f"[CANDIDATE] url={norm_url} snippet_words={snippet_words} snippet='{snippet_preview}'")

                # Gate 3: snippet richness. Exception: thin snippet OK if Tavily
                # already fetched substantive raw content for this URL.
                if snippet_words < MIN_SNIPPET_WORDS:
                    tavily_content = getattr(r, "content", None) or ""
                    if len(tavily_content.split()) < MIN_CONTENT_WORDS:
                        print(f"[FILTER:gate3:snippet] DROPPED url={norm_url} snippet_words={snippet_words}")
                        step_filtered += 1
                        filtered_count += 1
                        continue
                    print(f"[FILTER:gate3:snippet] KEPT url={norm_url} thin_snippet_but_tavily_content_ok")

                domain = urlparse(norm_url).netloc

                domain_failures = failed_domains.get(domain, 0)
                if domain_failures >= DOMAIN_FAIL_THRESHOLD:
                    print(f"[FILTER:gate4:runtime] DROPPED url={norm_url} domain={domain} failures={domain_failures}")
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

            phase1[step_id] = {
                "query": query,
                "candidates": unique_results,
                "raw_len": len(raw_results),
                "pre_filtered_len": len(pre_filtered),
                "deduped_len": len(deduped_results),
                "step_filtered": step_filtered,
            }
            print(f"[STEP {step_id}] Phase-1 candidates (pre-curation): {len(unique_results)}")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2 — Per-step LLM curation + scrape
        # Curation uses Tier 0 (auto-accept Tavily-rich sources) so most
        # steps only pay LLM cost for the small ambiguous remainder.
        # ═══════════════════════════════════════════════════════════════
        for step in steps:
            if getattr(state, "api_failure", False):
                break

            step_id = step.step_id
            p1 = phase1.get(step_id, {})
            query = p1.get("query", step.question)
            step_filtered = p1.get("step_filtered", 0)

            if step_status.get(step_id) != "success":
                continue

            pre_curation_count = len(p1.get("candidates", []))
            print(f"[STEP {step_id}] Pre-curation candidates: {pre_curation_count}")
            unique_results = _curate_sources(p1.get("candidates", []), query)
            print(f"[STEP {step_id}] Post-curation: {len(unique_results)}/{pre_curation_count}")

            # Score each candidate once, then sort+slice to step budget.
            # Caching the score avoids a second _pre_scrape_score pass inside
            # the scrape loop (score computation involves content parsing).
            scored = [
                (r, norm_url, _pre_scrape_score(r, query))
                for r, norm_url in unique_results
            ]
            scored.sort(key=lambda x: x[2], reverse=True)
            scored = scored[:MAX_RESULTS_PER_STEP]

            structured_results = []
            step_snippet_fallbacks = 0

            for i, (r, norm_url, initial_score) in enumerate(scored):
                domain = urlparse(norm_url).netloc

                print(f"\n[SCRAPE LOOP] rank={i+1}/{len(scored)} pre_scrape={initial_score:.3f} url={norm_url}")

                title = getattr(r, "title", "Untitled")
                snippet = getattr(r, "snippet", "") or title
                citation_counter += 1
                citation_id = f"[{citation_counter}]"

                # ── Content acquisition + citation status ──
                # Tavily pre-fetched → local scrape → snippet-only
                # Status derived from whichever HTTP path actually fires; never double-fetch.
                tavily_content = getattr(r, "content", None)
                content = tavily_content if tavily_content else None
                publish_date = None
                status: CitationStatus = CitationStatus.valid

                if content:
                    print(f"[CONTENT:tavily] url={norm_url} words={len(content.split())} → Tavily pre-fetched")

                elif i < MAX_SCRAPES and initial_score >= SCRAPE_MIN_RELEVANCE:
                    print(f"[CONTENT:scrape] url={norm_url} rank={i+1} score={initial_score:.3f} → local scrape")
                    try:
                        scraped = scrape_url(norm_url)
                        scrape_status = scraped.get("status", "failed")
                        error_type = scraped.get("error_type")

                        if scrape_status == "success":
                            content = scraped.get("content")
                            publish_date = scraped.get("publish_date")
                            status = CitationStatus.valid
                            print(f"[CONTENT:scrape:SUCCESS] url={norm_url} words={len((content or '').split())}")

                        elif scrape_status == "low_content":
                            status = CitationStatus.valid
                            print(f"[CONTENT:scrape:LOW_CONTENT] url={norm_url} → snippet fallback")
                            state.errors.append(ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.parsing_error,
                                message=f"[QUALITY] Low content url={norm_url} → snippet fallback"
                            ))
                            step_snippet_fallbacks += 1
                            snippet_fallback_count += 1

                        else:
                            state.failure_counts["parsing_failures"] += 1
                            status = status_from_scrape(error_type)
                            print(f"[CONTENT:scrape:FAILED] url={norm_url} error_type={error_type} → snippet fallback")
                            state.errors.append(ErrorLog(
                                node="searcher_node",
                                timestamp=datetime.now().isoformat(),
                                severity=SeverityEnum.WARNING,
                                error_type=ErrorTypeEnum.system_error,
                                message=f"[SCRAPER ERROR] url={norm_url} error_type={error_type} → snippet fallback"
                            ))
                            # `non_article` is a page-level Trafilatura verdict
                            # (one URL happened to be a listing/stub), NOT a
                            # domain-level signal — e.g. bbc.com/sport/index is
                            # non-article but bbc.com publishes great articles.
                            # Only count errors that plausibly repeat across
                            # the whole domain (auth walls, DNS/timeouts,
                            # servers consistently serving non-HTML).
                            if error_type in {
                                "auth_blocked", "timeout_error", "network_error",
                                "non_html_content",
                            }:
                                failed_domains[domain] = failed_domains.get(domain, 0) + 1
                                print(f"[RUNTIME:domain_fail] domain={domain} failures={failed_domains[domain]} (threshold={DOMAIN_FAIL_THRESHOLD})")
                            step_snippet_fallbacks += 1
                            snippet_fallback_count += 1

                    except Exception as e:
                        state.failure_counts["parsing_failures"] += 1
                        status = CitationStatus.stale
                        print(f"[CONTENT:scrape:EXCEPTION] url={norm_url} error={e} → snippet fallback")
                        state.errors.append(ErrorLog(
                            node="searcher_node",
                            timestamp=datetime.now().isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.system_error,
                            message=f"[SCRAPER ERROR] url={norm_url} reason={str(e)} → snippet fallback"
                        ))
                        step_snippet_fallbacks += 1
                        snippet_fallback_count += 1

                else:
                    # Snippet-only: no HTTP hit yet, so we need a reachability probe.
                    try:
                        status = validate_url(norm_url)
                    except Exception as e:
                        state.failure_counts["parsing_failures"] += 1
                        status = CitationStatus.broken
                        state.errors.append(ErrorLog(
                            node="searcher_node",
                            timestamp=datetime.now().isoformat(),
                            severity=SeverityEnum.WARNING,
                            error_type=ErrorTypeEnum.system_error,
                            message=f"URL validation failed: {norm_url} | {str(e)}"
                        ))
                    reason = (
                        f"rank_limit (rank={i+1} > MAX_SCRAPES={MAX_SCRAPES})"
                        if i >= MAX_SCRAPES
                        else f"low_relevance (score={initial_score:.3f} < SCRAPE_MIN_RELEVANCE={SCRAPE_MIN_RELEVANCE})"
                    )
                    print(f"[CONTENT:snippet_only] url={norm_url} reason={reason}")

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

            per_step_metrics[step_id] = {
                "raw": p1.get("raw_len", 0),
                "after_structural": p1.get("pre_filtered_len", 0),
                "after_dedup": p1.get("deduped_len", 0),
                "pre_curation": pre_curation_count,
                "post_curation": len(unique_results),
                "filtered": step_filtered,
                "snippet_fallbacks": step_snippet_fallbacks,
                "final": len(structured_results),
            }

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

    # -----------------------------------------------
    # OBSERVABILITY
    # -----------------------------------------------
    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()}
    )

    # Persist runtime-learned domain failures so supervisor re-search loops
    # don't rediscover the same unusable domains.
    state.failed_domains = failed_domains

    existing_log = state.node_logs.get(NodeNames.SEARCHER, {})
    existing_log.update({
        "total_steps": len(state.research_plan),
        "results_per_step": {k: len(v) for k, v in state.search_results.items()},
        "per_step_metrics": per_step_metrics,
        "step_status": step_status,
        "successful_steps": sum(1 for s in step_status.values() if s == "success"),
        "failed_steps": sum(1 for s in step_status.values() if s == "failed"),
        "total_sources": len(state.citations),
        "search_failures": state.failure_counts["search_failures"],
        "parsing_failures": state.failure_counts["parsing_failures"],
        "urls_filtered": filtered_count,
        "snippet_fallbacks": snippet_fallback_count,
        "learned_failed_domains": dict(failed_domains),
        "errors_count": len(state.errors)
    })
    state.node_logs[NodeNames.SEARCHER] = existing_log

    return state
