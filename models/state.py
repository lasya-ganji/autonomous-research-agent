from pydantic import BaseModel, Field
from typing import List, Dict, Set, Any, Optional

from models.planner_models import PlanStep
from models.search_models import SearchResult
from models.evaluation_models import EvaluationResult
from models.citation_models import Citation
from models.synthesis_models import SynthesisModel
from models.report_models import ReportModel
from models.error_models import ErrorLog
from models.cache_models import CacheModel

class ResearchState(BaseModel):
    # INPUT
    query: str

    # PLANNER
    research_plan: List[PlanStep] = Field(default_factory=list)

    # SEARCH
    search_results: Dict[int, List[SearchResult]] = Field(default_factory=dict)
    failed_queries: List[str] = Field(default_factory=list)

    # DEDUPLICATION
    deduplicated_urls: Set[str] = Field(default_factory=set)

    # CONTEXT BUILDING
    context_docs: List[str] = Field(default_factory=list)
    doc_summaries: Dict[str, str] = Field(default_factory=dict)

    # EVALUATION
    evaluation: Optional[EvaluationResult] = None
    failure_reason: str = ""
    overall_confidence: float = 0.0  

    # RETRY / REPLAN
    search_retry_count: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)

    # EXECUTION TRACKING
    unresolved_steps: List[int] = Field(default_factory=list)
    node_execution_count: int = Field(default=0, ge=0, le=12)

    # OUTPUT
    synthesis: Optional[SynthesisModel] = None
    report: Optional[ReportModel] = None

    # CITATIONS (SOURCE OF TRUTH)
    citations: Dict[str, Citation] = Field(default_factory=dict)

    # Track which citations are used in synthesis/report
    used_citation_ids: Set[str] = Field(default_factory=set)
    citation_mapping: Dict[str, str] = {}

    # BUDGET / CACHING
    budget_remaining: int = Field(default=0, ge=0)
    cache_hit: bool = False
    caches: List[CacheModel] = Field(default_factory=list)

    # ERRORS
    errors: List[ErrorLog] = Field(default_factory=list)

    # DEBUG / OBSERVABILITY
    node_logs: Dict[str, Any] = Field(default_factory=dict)

    total_tokens: int = 0
    total_cost: float = 0.0
    cost_limit: float = 2.0   # ₹ limit (change if needed)
    abort: bool = False

    