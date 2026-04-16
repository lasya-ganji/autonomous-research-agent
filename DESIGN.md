# 1. Architecture Diagram

The agent is implemented as a LangGraph `StateGraph` where each node operates on a shared `ResearchState`. The **Supervisor** acts as the central routing hub controlling execution flow.

```mermaid
graph TD
    START([User Query]) --> SUP[Supervisor]

    SUP -->|first run| PL[Planner]
    SUP -->|retry| SE[Searcher]
    SUP -->|replan| PL
    SUP -->|proceed| SY[Synthesiser]
    SUP -->|abort / max exec / api_failure| RE[Reporter]

    PL --> SE
    SE --> EV[Evaluator]
    EV --> SUP

    SY --> CM[Citation Manager]
    CM --> RE
    RE --> END([Final Report])
```

---

## Node Responsibilities

| Node | Responsibility |
|------|---------------|
| Supervisor | Controls execution flow, enforces limits, routes decisions |
| Planner | Decomposes query into ≤3 sub-questions |
| Searcher | Retrieves and processes sources |
| Evaluator | Scores results and decides next step |
| Synthesiser | Generates claims from sources |
| Citation Manager | Verifies claims against sources |
| Reporter | Produces final structured report |

---

# 2. State Schema

All nodes share a single `ResearchState`.

---

## Input

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| query | str | User query | Entry point for the system |

---

## Planner

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| research_plan | List[PlanStep] | Sub-questions | Breaks complex query into manageable steps |

---

## Search

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| search_results | Dict[int, List[SearchResult]] | Results per step | Enables step-wise evaluation |

---

## Evaluation

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| evaluation | Optional[EvaluationResult] | Evaluation output | None on first run → triggers planner |
| overall_confidence | float | Confidence score | Drives routing decisions |
| failure_reason | str | Failure explanation | Helps planner during replan |

---

## Retry / Replan

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| search_retry_count | int | Retry attempts | Prevents infinite retry loops |
| replan_count | int | Replan attempts | Prevents infinite replanning |

---

## Execution

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| node_execution_count | int | Total executions | Enforces execution cap (12) |
| unresolved_steps | List[int] | Failed steps | Tracks incomplete coverage |

---

## Output

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| synthesis | Optional[SynthesisModel] | Generated claims | Intermediate output |
| report | Optional[ReportModel] | Final report | End result |

---

## Citations

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| citations | Dict[str, Citation] | Citation registry | Single source of truth |
| used_citation_ids | Set[str] | Used citations | Ensures clean output |
| citation_mapping | Dict[str, str] | ID mapping | User-friendly numbering |
| citation_chunks | Dict[str, List[str]] | Source chunks | Used for verification |

---

## Failure Tracking

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| failure_counts | Dict[str, int] | Tracks failures | Used in routing decisions |

---

## Control Flags

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| is_partial | bool | Partial output flag | Indicates incomplete execution |
| api_failure | bool | API failure flag | Triggers early termination |
| abort | bool | Cost abort flag | Stops execution if budget exceeded |

---

## Observability

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| errors | List[ErrorLog] | Error logs | Debugging and traceability |
| node_logs | Dict[str, Any] | Node logs | Observability |
| next_node | Optional[str] | Routing target | Used by LangGraph |

---

## Cost Tracking

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| total_tokens | int | Tokens used | Observability |
| total_cost | float | Cost (INR) | Budget tracking |
| cost_limit | float | Max cost | Prevents excessive spend |

---

## Latency

| Field | Type | Description | Rationale |
|------|------|------------|-----------|
| start_time | float | Start time | Measure latency |
| elapsed_time | float | Runtime | Performance tracking |

---

# 3. Tool Integration

| Tool | Purpose | Input | Output | Failure Handling |
|------|--------|------|--------|-----------------|
| call_llm | LLM completion via GPT-4o-mini | prompt: str, temperature: float | {content, usage} | Returns empty + error |
| search_tool | Web search via Tavily | query: str | List[SearchResult] OR error dict | Structured error response |
| scrape_url | Full-page content extraction | url: str | {content, publish_date} | Trafilatura → BeautifulSoup fallback |
| get_embedding | Local embedding model | text: str | List[float] | Returns None on failure |
| validate_url | URL reachability check | url: str | CitationStatus | Handles HTTP errors |
| get_dynamic_weights | LLM-based scoring weights | query: str, state | Dict[str, float] | Falls back to defaults |

---

# 4. Citation Design

## Citation Schema

```python
class Citation(BaseModel):
    citation_id: str
    url: HttpUrl
    title: str
    author: Optional[str]
    date_accessed: str
    quality_score: float
    status: CitationStatus

class Claim(BaseModel):
    text: str
    citation_ids: List[str]
    confidence: float
    verified: bool
    citation_confidence: float
    citation_score_map: Dict[str, float]
    hallucinated_citations: List[str]
```

## Hallucination Detection

A citation is considered hallucinated if:
- No content available  
- URL invalid  
- similarity score ≤ 0.4  

A claim is unverified if:
- no valid citations  
- OR average similarity ≤ 0.5  

---

## Design Approach

- Multi-stage filtering:
  1. Search filtering  
  2. Synthesis selection  
  3. Citation verification  

Only verified citations appear in final output.

---

# 5. Challenges

## 1. Replanning Loop Prevention

**Problem:**  
Unanswerable queries can cause infinite retry/replan loops.

**Solution:**  
- MAX_SEARCH_RETRIES = 1  
- MAX_REPLANS = 1  
- MAX_NODE_EXECUTIONS = 12  
- Supervisor forces proceed after limits  

**Limitation:**  
May terminate early for complex queries.

---

## 2. Citation Quality vs Coverage

**Problem:**  
Too many sources reduce quality, too few reduce coverage.

**Solution:**  
- Filtering at search stage  
- Top-k selection in synthesis  
- Final verification in citation manager  

**Limitation:**  
Some useful low-quality sources may be dropped.

---

## 3. Contradictory Sources

**Problem:**  
Different sources may provide conflicting information.

**Solution:**  
- The synthesiser prompts the LLM to explicitly identify conflicts and return them as a `conflicts` array in the structured output.  
- The Claim model supports multiple `citation_ids`, allowing conflicting sources to be cited together.  

**Limitation:**  
Conflict detection is LLM-dependent and may miss subtle contradictions or numerical inconsistencies.

---

## 4. Stateless System

**Problem:**  
The system has no memory across runs, leading to repeated work.

**Solution:**  
- Within a run, deduplication is handled using a multi-stage pipeline (URL, title, semantic).  
- `search_results` is keyed by `step_id`, enabling step-wise tracking and evaluation.  

**Limitation:**  
No cross-session memory — repeated queries are processed from scratch.

---

## 5. Cost vs Quality Tradeoff

**Problem:**  
More processing improves quality but increases cost.

**Solution:**  
- cost_limit enforcement  
- execution limit  
- partial output fallback  

**Limitation:**  
May stop before full coverage.

---

# 6. Cost Estimates

| Scenario | Tokens | Cost |
|----------|--------|------|
| Normal | ~7400 | ₹0.13 |
| Retry | ~9000 | ₹0.16 |
| Replan | ~10500 | ₹0.18 |
| Worst | ~13000 | ₹0.22 |