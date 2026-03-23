from pydantic import BaseModel, Field
from typing import Dict


class ScoringWeights(BaseModel):
    relevance: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    domain: float = Field(ge=0, le=1)
    depth: float = Field(ge=0, le=1)


class ScoreBreakdown(BaseModel):
    relevance: float = Field(ge=0, le=1)
    domain: float = Field(ge=0, le=1)
    recency: float = Field(ge=0, le=1)
    depth: float = Field(ge=0, le=1)


class DocumentScore(BaseModel):
    citation_id: str
    final_score: float = Field(ge=0, le=1)
    breakdown: ScoreBreakdown


class ConfidenceMetrics(BaseModel):
    average_score: float = Field(ge=0, le=1)
    agreement: float = Field(ge=0, le=1)
    consistency: float = Field(ge=0, le=1)
    top_k_margin: float = Field(ge=0, le=1)


class EvaluationResult(BaseModel):
    confidence_score: float = Field(ge=0, le=1)
    decision: str  # retry | replan | proceed
    weights: ScoringWeights
    metrics: ConfidenceMetrics