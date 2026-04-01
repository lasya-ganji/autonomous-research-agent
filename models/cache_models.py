from pydantic import BaseModel, Field
from typing import List, Any

class CacheModel(BaseModel):
    query_text: str
    embedding: List[float]
    result: Any
    timestamp: str
    ttl_seconds: int = Field(gt=0)
    last_accessed: str
    quality_score: float = Field(ge=0, le=1)
    citation_confidence: float = Field(ge=0, le=1)
    is_failed: bool = False
    
