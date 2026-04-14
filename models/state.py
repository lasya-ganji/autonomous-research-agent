from pydantic import BaseModel, Field
from typing import List, Dict, Set, Any, Optional

from models.planner_models import PlanStep
from models.search_models import SearchResult
from models.evaluation_models import EvaluationResult
from models.citation_models import Citation
from models.synthesis_models import SynthesisModel
from models.report_models import ReportModel
from models.error_models import ErrorLog


class ResearchState(BaseModel):

    # INPUT
    query: str

    # PLANNER
    research_plan: List[PlanStep] = Field(default_factory=list)

    # SEARCH
    search_results: Dict[int, List[SearchResult]] = Field(default_factory=dict)
    

    # DEDUP
    #deduplicated_urls: Set[str] = Field(default_factory=set)

    # CONTEXT
    #context_docs: List[str] = Field(default_factory=list)
    

    # EVALUATION
    evaluation: Optional[EvaluationResult] = None
    failure_reason: str = ""
    overall_confidence: float = 0.0

    # RETRY / REPLAN
    search_retry_count: int = Field(default=0, ge=0)
    replan_count: int = Field(default=0, ge=0)

    # EXECUTION
    unresolved_steps: List[int] = Field(default_factory=list)
    node_execution_count: int = Field(default=0, ge=0)

    # OUTPUT
    synthesis: Optional[SynthesisModel] = None
    report: Optional[ReportModel] = None

    # CITATIONS
    citations: Dict[str, Citation] = Field(default_factory=dict)
    used_citation_ids: Set[str] = Field(default_factory=set)
    citation_mapping: Dict[str, str] = Field(default_factory=dict)
    citation_chunks: Dict[str, List[str]] = Field(default_factory=dict)

    failure_counts: Dict[str, int] = Field(default_factory=lambda: {
        "search_failures": 0,
        "parsing_failures": 0,
        "low_confidence": 0,
        "citation_failures": 0,
    })

    is_partial: bool = False

    # ERRORS
    errors: List[ErrorLog] = Field(default_factory=list)

    # OBSERVABILITY
    node_logs: Dict[str, Any] = Field(default_factory=dict)

    total_tokens: int = 0
    total_cost: float = 0.0
    cost_limit: float = 2.0
    abort: bool = False

    # ROUTING
    next_node: Optional[str] = None
    
    # TIME TRACKING
    start_time: Optional[float] = None
    elapsed_time: float = 0.0