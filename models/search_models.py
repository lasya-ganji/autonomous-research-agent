from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class SearchResult(BaseModel):
    citation_id: str
    url: HttpUrl
    title: str
    snippet: str
    content: Optional[str] = None

    quality_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    domain_score: float = Field(ge=0, le=1)
    depth_score: float = Field(ge=0, le=1)

    rank: int = Field(ge=1)