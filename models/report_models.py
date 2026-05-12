from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ReportModel(BaseModel):
    title: str
    sections: List[str]
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    claim_evidence: List[Dict[str, Any]] = Field(default_factory=list)