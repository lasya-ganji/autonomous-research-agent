from pydantic import BaseModel, Field
from typing import List, Dict, Set

from models.enums import DecisionEnum
from models.planner_models import PlanStep
from models.search_models import SearchResult
from models.citation_models import Citation
from models.synthesis_models import SynthesisModel
from models.report_models import ReportModel
from models.error_models import ErrorLog
from models.cache_models import CacheModel


class Evaluation(BaseModel):
    confidence_score: float = Field(ge=0, le=1, default=0.0)
    decision: DecisionEnum = DecisionEnum.proceed


class ResearchState(BaseModel):
    query: str

    research_plan: List[PlanStep] = Field(default_factory=list)

    search_results: Dict[int, List[SearchResult]] = Field(default_factory=dict)

    citations: Dict[str, Citation] = Field(default_factory=dict)

    deduplicated_urls: Set[str] = Field(default_factory=set)

    context_docs: List[str] = Field(default_factory=list)

    doc_summaries: Dict[str, str] = Field(default_factory=dict)

    synthesis: SynthesisModel | None = None
    report: ReportModel | None = None

    evaluation: Evaluation = Evaluation()

    search_retry_count: int = Field(ge=0, default=0)
    replan_count: int = Field(ge=0, default=0)

    node_execution_count: int = Field(ge=0, le=12, default=0)

    token_usage: int = Field(ge=0, default=0)
    budget_remaining: int = Field(ge=0, default=0)

    cache_hit: bool = False

    unresolved_steps: List[int] = Field(default_factory=list)

    errors: List[ErrorLog] = Field(default_factory=list)

    caches: List[CacheModel] = Field(default_factory=list)
    failed_queries: List[str] = Field(default_factory=list)