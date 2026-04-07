from pydantic import BaseModel, Field
from typing import List, Dict

class Claim(BaseModel):
    text: str
    citation_ids: List[str]
    confidence: float = Field(ge=0, le=1)
    verified: bool
    citation_confidence: float = Field(ge=0, le=1)
    citation_score_map: Dict[str, float] = {}
    hallucinated_citations: List[str] = []

class Conflict(BaseModel):
    claim_a: str
    claim_b: str
    citations_a: List[str]
    citations_b: List[str]


class SynthesisModel(BaseModel):
    claims: List[Claim] = []
    conflicts: List[Conflict] = []
    partial: bool = False