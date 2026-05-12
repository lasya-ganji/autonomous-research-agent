from pydantic import BaseModel, Field
from typing import List, Dict


class ClaimEvidence(BaseModel):
    citation_id: str
    source_title: str = ""
    source_url: str = ""
    evidence_snippet: str = ""
    support_score: float = Field(default=0.0, ge=0, le=1)
    matches_claim: bool = False

class Claim(BaseModel):
    text: str
    citation_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    verified: bool = False
    citation_confidence: float = Field(default=0.0, ge=0, le=1)
    citation_score_map: Dict[str, float] = Field(default_factory=dict)
    hallucinated_citations: List[str] = Field(default_factory=list)
    support_status: str = "partially_verified"
    support_reason: str = "citation_support_pending"
    evidence: List[ClaimEvidence] = Field(default_factory=list)

class Conflict(BaseModel):
    claim_a: str
    claim_b: str
    citations_a: List[str]
    citations_b: List[str]


class SynthesisModel(BaseModel):
    claims: List[Claim] = []
    conflicts: List[Conflict] = []
    partial: bool = False