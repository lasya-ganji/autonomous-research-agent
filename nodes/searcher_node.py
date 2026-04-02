from models.state import ResearchState
from models.search_models import SearchResult

from tools.search_tool import search_tool as real_search_tool
from tools.scraper_tool import scrape_url

from utils.logger import log_node_execution
from observability.tracing import trace_node

from app.dependencies import semantic_cache
from services.retrieval.embedding_service import get_embedding

import time
import uuid


@trace_node("searcher_node")
def safe_search(query, search_tool_func):
    for i in range(3):   # retries
        try:
            results = search_tool_func.search(query)  

            if results:
                return results

        except Exception as e:
            print(f"[RETRY {i+1}] Search failed:", e)

        time.sleep(2 ** i)

    print("[SEARCH FAILED] Returning empty results")
    return []


def searcher_node(state: ResearchState, search_tool=None) -> ResearchState:
    print("Searcher Node")

    # ✅ choose correct search tool (mock or real)
    actual_search_tool = search_tool if search_tool else real_search_tool

    if getattr(state, "skip_search", False):
        print("⚡ Skipping search (cache hit)")
        return state

    start_time = time.time()

    # CACHE CHECK
    try:
        normalized_query = state.query.strip().lower()
        query_embedding = get_embedding(normalized_query)

        if query_embedding is None:
            print("[CACHE SKIP] No embedding")
            state.cache_hit = False
            cached = None
        else:
            cached = semantic_cache.search(query_embedding)

        if cached:
            print("🔥 CACHE HIT")

            state.cache_hit = True
            state.report = cached.result
            state.overall_confidence = cached.citation_confidence

            state.skip_search = True
            state.skip_eval = True
            state.skip_remaining = True

            if state.node_logs is None:
                state.node_logs = {}

            state.node_logs["cache"] = {
                "hit": True,
                "quality": cached.quality_score,
                "confidence": cached.citation_confidence
            }

            return state

        print("❌ CACHE MISS")
        state.cache_hit = False

    except Exception as e:
        print(f"[CACHE ERROR] {e}")
        state.cache_hit = False

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    state.search_results = {}

    try:
        steps = sorted(state.research_plan, key=lambda x: x.priority)

        for step in steps:
            step_id = step.step_id
            base_query = step.question

            if state.search_retry_count > 0:
                query = f"{base_query} detailed explanation examples latest"
            else:
                query = base_query

            print(f"[SEARCHER NODE] Step {step_id}: {query}")

            # ✅ FIXED: pass correct tool
            raw_results = safe_search(query, actual_search_tool)

            structured_results = []
            seen_urls = set()

            scrape_limits = {1: 3, 2: 2, 3: 1}
            max_scrapes = scrape_limits.get(step.priority, 1)

            if raw_results:
                for i, r in enumerate(raw_results):

                    # ✅ handle both dict (test) and object (real)
                    if isinstance(r, dict):
                        url = r.get("url")
                        title = r.get("title", "Untitled")
                        snippet = r.get("snippet", "")
                    else:
                        url = getattr(r, "url", None)
                        title = getattr(r, "title", "Untitled")
                        snippet = getattr(r, "snippet", "")

                    if not url or url in seen_urls:
                        continue

                    seen_urls.add(url)
                    citation_id = str(uuid.uuid4())

                    content = None
                    if i < max_scrapes:
                        try:
                            content = scrape_url(url)
                        except Exception as e:
                            print(f"[SCRAPER ERROR] {url}: {e}")

                    if not snippet:
                        snippet = title

                    result = SearchResult(
                        citation_id=citation_id,
                        url=url,
                        title=title,
                        snippet=snippet,
                        content=content,
                        quality_score=0.0,
                        relevance_score=0.0,
                        recency_score=0.0,
                        domain_score=0.0,
                        depth_score=0.0,
                        rank=1
                    )

                    structured_results.append(result)

            state.search_results[step_id] = structured_results

    except Exception as e:
        print(f"[SEARCHER NODE ERROR] {e}")

    log_node_execution(
        node_name="searcher_node",
        input_data=state.query,
        output_data={k: len(v) for k, v in state.search_results.items()},
        start_time=start_time
    )

    if state.node_logs is None:
        state.node_logs = {}

    state.node_logs["searcher"] = {
        "total_steps": len(state.research_plan),
        "results_per_step": {
            step_id: len(results)
            for step_id, results in state.search_results.items()
        },
        "retry_mode": state.search_retry_count > 0
    }

    state.node_execution_count += 1

    return state