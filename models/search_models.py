from pydantic import BaseModel, HttpUrl
from typing import Optional

class SearchResult(BaseModel):
    citation_id: str
    url: HttpUrl
    title: str
    snippet: str
    content: Optional[str] = None

    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    recency_score: Optional[float] = None
    domain_score: Optional[float] = None
    depth_score: Optional[float] = None

    rank: Optional[int] = None