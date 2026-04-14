from typing import List, Union
from tavily import TavilyClient
from models.search_models import SearchResult
import uuid
import os
from dotenv import load_dotenv
from config.constants.node_constants.search_constants import TAVILY_MAX_RESULTS

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_tool(query: str) -> Union[List[SearchResult], dict]:
    print(f"[SEARCH TOOL] Query: {query}")

    try:
        response = client.search(query=query, max_results=TAVILY_MAX_RESULTS)

        results: List[SearchResult] = []

        for i, item in enumerate(response.get("results", [])):
            result = SearchResult(
                citation_id=str(uuid.uuid4()),
                url=item.get("url"),
                title=item.get("title", ""),
                snippet=item.get("content", "") or "",
                content=None,
                quality_score=0.5,
                relevance_score=0.5,
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
