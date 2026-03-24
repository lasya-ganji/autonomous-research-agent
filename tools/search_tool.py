from typing import List
from tavily import TavilyClient
from models.search_models import SearchResult
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_tool(query: str) -> List[SearchResult]:
    print(f"[SEARCH TOOL] Query: {query}")

    try:
        response = client.search(
            query=query,
            max_results=5
        )

        results: List[SearchResult] = []

        for i, item in enumerate(response.get("results", [])):
            result = SearchResult(
                citation_id=str(uuid.uuid4()),
                url=item.get("url"),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                content=None,

                # neutral scores (scoring_service will handle later)
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
        print(f"[SEARCH TOOL ERROR] {e}")
        return []