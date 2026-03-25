from pydantic import BaseModel
from typing import List


class StepEvaluation(BaseModel):
    step_id: int
    confidence_score: float
    passed: bool


class EvaluationResult(BaseModel):
    steps: List[StepEvaluation]
    decision: str  # "proceed" | "retry" | "replan"