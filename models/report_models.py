from pydantic import BaseModel
from typing import List, Dict, Any

class ReportModel(BaseModel):
    title: str
    sections: List[str]
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]