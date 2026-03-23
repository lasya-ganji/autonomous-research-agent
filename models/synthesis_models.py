from pydantic import BaseModel, Field
from typing import List

class Claim(BaseModel):
    text: str
    citation_ids: List[str]
    confidence: float = Field(ge=0, le=1)
    verified: bool
    citation_confidence: float = Field(ge=0, le=1)


class Conflict(BaseModel):
    claim_a: str
    claim_b: str
    citations_a: List[str]
    citations_b: List[str]


class SynthesisModel(BaseModel):
    claims: List[Claim] = []
    conflicts: List[Conflict] = []
    partial: bool = False