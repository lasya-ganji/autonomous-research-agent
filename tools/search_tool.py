from typing import List, Union
from tavily import TavilyClient
from models.search_models import SearchResult
import uuid
import os
from dotenv import load_dotenv
from config.constants.node_constants.search_constants import TAVILY_MAX_RESULTS
from config.constants.scraper_constants import MIN_CONTENT_WORDS, MAX_CONTENT_CHARS, MIN_TAVILY_CONTENT_SCORE, MIN_RESULT_SCORE

def get_tavily_client():
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY")

    return TavilyClient(api_key=api_key)


def search_tool(query: str) -> Union[List[SearchResult], dict]:
    print(f"[SEARCH TOOL] Query: {query}")

    try:
        client = get_tavily_client()
        response = client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            include_raw_content=True,
        )

        results: List[SearchResult] = []

        for i, item in enumerate(response.get("results", [])):
            # Use Tavily's server-side extracted content when available.
            # Tavily fetches from their whitelisted IP pool, bypassing 403s that
            # our own scraper hits. Fall back to None so searcher_node scrapes itself.
            #
            # Gate 0: drop results that are clearly off-topic before they enter the
            # pipeline. Score < MIN_RESULT_SCORE (e.g. 0.09 for a retirees page)
            # means Tavily itself rates this as a low-quality match — including it
            # drags avg quality down without contributing useful information.
            #
            # Gate 1 (raw_content): Tavily relevance score >= MIN_TAVILY_CONTENT_SCORE.
            # Gate 2 (raw_content): Char cap before word-count to prevent enormous pages.
            tavily_score = float(item.get("score", 0.0))

            if tavily_score < MIN_RESULT_SCORE:
                print(f"[SCORE GATE] Dropped url={item.get('url')} score={tavily_score:.3f} (below MIN_RESULT_SCORE={MIN_RESULT_SCORE})")
                continue
            tavily_content = (item.get("raw_content") or "").strip()[:MAX_CONTENT_CHARS * 6]
            pre_fetched_content = (
                tavily_content
                if tavily_score >= MIN_TAVILY_CONTENT_SCORE and len(tavily_content.split()) >= MIN_CONTENT_WORDS
                else None
            )

            result = SearchResult(
                citation_id=str(uuid.uuid4()),
                url=item.get("url"),
                title=item.get("title", ""),
                snippet=item.get("content", "") or "",
                content=pre_fetched_content,
                quality_score=0.5,
                relevance_score=float(item.get("score", 0.5)),  # Tavily ML score
                recency_score=0.5,
                domain_score=0.5,
                depth_score=0.5,
                rank=i + 1
            )
            results.append(result)

        print(f"[SEARCH TOOL] Results fetched: {len(results)}")
        return results

    except Exception as e:
        error_msg = str(e).lower()

        # -------------------------------
        # ERROR CLASSIFICATION
        # -------------------------------
        if any(x in error_msg for x in ["unauthorized", "invalid api key", "401"]):
            error_type = "api_error"
            severity = "CRITICAL"
            retryable = False

        elif any(x in error_msg for x in ["timeout", "timed out"]):
            error_type = "timeout_error"
            severity = "WARNING"
            retryable = True

        elif any(x in error_msg for x in ["connection", "network", "dns"]):
            error_type = "network_error"
            severity = "WARNING"
            retryable = True

        else:
            error_type = "unknown_error"
            severity = "WARNING"
            retryable = True

        print(f"[SEARCH TOOL ERROR] {e} | TYPE: {error_type}")

        # -------------------------------
        # RETURN STRUCTURED ERROR
        # -------------------------------
        return {
            "error": str(e),
            "type": error_type,
            "severity": severity,
            "retryable": retryable
        }
