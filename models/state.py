from pydantic import BaseModel, Field
from typing import List, Dict, Set, Any

from models.planner_models import PlanStep
from models.search_models import SearchResult
from models.evaluation_models import EvaluationResult
from models.citation_models import Citation
from models.synthesis_models import SynthesisModel
from models.report_models import ReportModel
from models.error_models import ErrorLog
from models.cache_models import CacheModel


class ResearchState(BaseModel):
    # Input
    query: str

    # Planner
    research_plan: List[PlanStep] = Field(default_factory=list)

    # Search
    search_results: Dict[int, List[SearchResult]] = Field(default_factory=dict)
    failed_queries: List[str] = Field(default_factory=list)

    # Deduplication
    deduplicated_urls: Set[str] = Field(default_factory=set)

    # Context building
    context_docs: List[str] = Field(default_factory=list)
    doc_summaries: Dict[str, str] = Field(default_factory=dict)

    # Evaluation
    evaluation: EvaluationResult | None = None
    failure_reason: str = ""
    overall_confidence: float = 0.0  

    # Retry / Replan control
    search_retry_count: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)

    # Execution tracking
    unresolved_steps: List[int] = Field(default_factory=list)
    node_execution_count: int = Field(default=0, ge=0, le=12)

    # Output
    synthesis: SynthesisModel | None = None
    report: ReportModel | None = None

    # Citations
    citations: Dict[str, Citation] = Field(default_factory=dict)

    # Budget / caching
    token_usage: int = Field(default=0, ge=0)
    budget_remaining: int = Field(default=0, ge=0)
    cache_hit: bool = False
    caches: List[CacheModel] = Field(default_factory=list)

    skip_search: bool = False
    skip_eval: bool = False
    skip_remaining: bool = False


    # Errors
    errors: List[ErrorLog] = Field(default_factory=list)

    # Observability / Debugging
    node_logs: Dict[str, Any] = Field(default_factory=dict)

