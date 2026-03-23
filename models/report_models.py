from pydantic import BaseModel
from typing import List, Dict

class ReportModel(BaseModel):
    title: str
    sections: List[str]
    citations: List[str]
    metadata: Dict