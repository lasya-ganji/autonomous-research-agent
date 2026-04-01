from pydantic import BaseModel, Field, HttpUrl
from typing import Optional
from models.enums import CitationStatus

class Citation(BaseModel):
    citation_id: str
    url: HttpUrl
    title: str
    author: Optional[str] = None
    date_accessed: str
    quality_score: float = Field(default=0.5, ge=0, le=1)
    status: CitationStatus = CitationStatus.valid