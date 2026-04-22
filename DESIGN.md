# 1. Architecture Diagram

The agent is implemented as a LangGraph `StateGraph` where each node receives and returns a `ResearchState` object. The **Supervisor** node acts as the central routing hub.

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
| Supervisor | Reads `evaluation.decision`, sets `next_node`, enforces execution budget |
| Planner | Decomposes query into ≤3 prioritised sub-questions via LLM |
| Searcher | Tavily search → 3-stage dedup → LLM curation → scrape → build citations |
| Evaluator | Score results → compute confidence → decide proceed / retry / replan / forced-proceed |
| Synthesiser | Build context chunks → LLM synthesis → extract claims |
| Citation Manager | Re-validate URLs → verify claims against source chunks |
| Reporter | Remap citation IDs → LLM report generation → build `ReportModel` |

---

# 2. State Schema

All nodes share a single `ResearchState` (Pydantic `BaseModel`) defined in `models/state.py`.

---

## Input

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| query | str | The raw user research question | Entry point |

---

## Planner

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| research_plan | List[PlanStep] | Up to 3 LLM-generated sub-questions with priority (1–5) | Decomposes the query into independently searchable units; sorted by priority; 3 selected to bound cost |

---

## Search

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| search_results | Dict[int, List[SearchResult]] | Maps step_id to scored results | Step-keyed so the evaluator can assess coverage per sub-question |

---

## Evaluation

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| evaluation | Optional[EvaluationResult] | Per-step evaluation and routing decision | None on first run — signals the supervisor to go to planner |
| failure_reason | str | Human-readable failure description | Fed back to planner replan prompt so the LLM can generate a different strategy |
| overall_confidence | float | Aggregate confidence score [0,1] | Decides whether to route to planner, searcher, or synthesiser |

---

## Retry / Replan

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| search_retry_count | int | Number of retry iterations attempted | Hard ceiling: `MAX_SEARCH_RETRIES = 1` |
| replan_count | int | Number of replanning iterations attempted | Hard ceiling: `MAX_REPLANS = 1` |

---

## Execution Tracking

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| node_execution_count | int | Total node executions across the run | Routes to reporter if count ≥ 12 with `is_partial = True` |
| unresolved_steps | List[int] | Step IDs that returned no results | Passed to replanner as context |

---

## Output

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| synthesis | Optional[SynthesisModel] | Claims and conflicts from LLM synthesis | Handles contradictory statements |
| report | Optional[ReportModel] | Final formatted report | Terminal output |

---

## Citations

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| citations | Dict[str, Citation] | Source of truth keyed by citation ID [N] | Single registry updated by searcher, re-validated by citation manager |
| used_citation_ids | Set[str] | IDs actually referenced in verified claims | Used by the reporter to build the final reference list |
| citation_mapping | Dict[str, str] | Internal IDs to sequential output IDs | Ensures clean numbering in the output report |
| citation_chunks | Dict[str, List[str]] | Scraped text chunks per citation ID | Used by citation manager for similarity-based hallucination detection |

---

## Failure Tracking

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| failure_counts | Dict[str, int] | Counts for search_failures, parsing_failures, low_confidence, citation_failures | Drives routing decisions in the supervisor |
| failed_domains | Dict[str, int] | Domains that failed scraping and their failure count | Passed to Tavily exclude_domains on subsequent searches |

---

## Cost Controls

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| total_tokens | int | Token count across all LLM calls | Observability |
| total_cost | float | Cost in INR | Checked against `cost_limit` at every LLM call |
| cost_limit | float | Default = ₹2.00 | Hard budget ceiling — sets `abort = True` if exceeded |
| abort | bool | Emergency exit flag | When True, supervisor bypasses all remaining nodes |

---

## Control Flags

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| is_partial | bool | Partial output flag | Indicates incomplete execution |
| api_failure | bool | API failure flag | Triggers early termination via supervisor |

---

## Errors & Observability

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| errors | List[ErrorLog] | Structured error log with node, severity, type, message | Debugging and traceability |
| node_logs | Dict[str, Any] | Per-node execution data including timing, tokens, cost | Surfaced in the Debug tab of the UI |
| next_node | Optional[str] | Target node set by supervisor | LangGraph reads this via `route_from_supervisor` |

---

## Latency

| Field | Type | Description | Rationale |
|-------|------|-------------|-----------|
| start_time | Optional[float] | Run start timestamp | Latency measurement |
| elapsed_time | float | Seconds elapsed since start | Used by supervisor for latency-aware routing (T1/T2/T3 thresholds) |

---

# 3. Tool Integration

| Tool | Purpose | Input Contract | Output Contract | Failure Handling |
|------|---------|---------------|----------------|-----------------|
| call_llm | Chat completion via GPT-4o-mini | `prompt: str`, `temperature: float` | `{"content": str, "usage": dict}` | Returns `"content": "", "usage": {}, "error": str` on exception |
| search_tool | Web search via Tavily | `query: str` | `List[SearchResult]` with relevance scores | Returns `[]` on any exception; searcher retries with fallback query |
| scrape_url | Full-page content extraction | `url: str` | `{"content": str, "publish_date": str\|None}` | Trafilatura failure falls back to BeautifulSoup; returns `low_content` or `failed` status |
| get_embedding | Local embedding model (all-MiniLM-L6-v2) | `text: str` | `List[float]` — 384 dimensions | Returns `None` on empty input or any exception |
| validate_url | HTTP reachability check | `url: str` | `CitationStatus` enum (valid / stale / broken) | Timeout → stale; 4xx → broken; 3xx → stale |
| get_dynamic_weights | LLM-driven scoring weights | `query: str`, `state` (optional) | `Dict[str, float]` with keys relevance, recency, domain, depth | Falls back to `DEFAULT_WEIGHTS = {relevance: 0.5, recency: 0.2, domain: 0.2, depth: 0.1}` on LLM failure |

---

# 4. Citation Schema and Design Rationale

```python
# models/citation_models.py
class Citation(BaseModel):
    citation_id: str          # Format: "[N]"
    url: HttpUrl              # Pydantic-validated URL
    title: str                # Page title from search result
    author: Optional[str]     # Extracted where available; often None
    date_accessed: str        # ISO date string
    quality_score: float      # [0, 1] — updated by scoring_service after evaluation
    status: CitationStatus    # valid | stale | broken

# models/synthesis_models.py
class Claim(BaseModel):
    text: str                                    # The factual statement
    citation_ids: List[str]                      # References to Citation registry
    confidence: float                            # LLM's self-reported confidence [0,1]
    verified: bool                               # True if citation_manager avg_score > 0.4
    citation_confidence: float                   # Average similarity score across citations
    citation_score_map: Dict[str, float]         # Per-citation similarity score
    hallucinated_citations: List[str]            # IDs that failed similarity check

class SynthesisModel(BaseModel):
    claims: List[Claim]
    conflicts: List[Conflict]   # Pairs of contradictory claims with their sources
    partial: bool               # True if synthesis fell back due to no valid data
```

## Design Rationale

**Two-registry pattern:** Citations are stored in `state.citations` (the source of truth) keyed by internal ID `[N]`. The reporter produces a second clean mapping (`state.citation_mapping`) that remaps these to sequential `[1]`, `[2]`... in the final output.

**Quality score propagation:** Citations begin with `quality_score = 0.5` (neutral). After the evaluator runs `scoring_service`, quality scores are backpropagated from `SearchResult` objects to their corresponding `Citation` entries. The citation manager then uses these, alongside semantic similarity, to make the final verification decision.

**Hallucination detection at the claim level:** Rather than trusting the LLM's self-reported citation IDs, the citation manager independently computes similarity between each claim's text and the actual scraped chunks for each cited source. A citation is considered hallucinated if it has no chunks, the URL is not reachable, or the best chunk similarity score is ≤ 0.3. A claim is marked `verified = False` if no citation passes or if the average score across passing citations is ≤ 0.4.

---

# 5. Known Limitations and Production Gaps

## Functional Limitations

**Synthesis context uses approximate chunk ranking:**
Chunks are ranked using word overlap rather than semantic embeddings. This is a CPU-only approximation that can miss semantically relevant chunks that share few exact words with the query.

**No persistent state or session memory:**
Each agent run starts with a fresh `ResearchState`. There is no caching of prior research, no cross-session memory, and no deduplication against previously researched topics. Every invocation is fully stateless.

**No parallelism across plan steps:**
The searcher node processes all 3 plan steps sequentially. Parallel execution using `asyncio` or `ThreadPoolExecutor` could reduce search latency by up to 3×.

**SentenceTransformer model loaded synchronously at import:**
`embedding_service.py` loads `all-MiniLM-L6-v2` at module import time. In a cold-start or serverless environment this adds 2–5 seconds of startup latency before the first request is served.

## Production Gaps

**No LLM fallback:** No fallback to an alternative provider if OpenAI is unavailable.

**No persistent storage:** Results are not saved — refreshing the UI loses all outputs.

**No async execution:** All I/O (scraping, LLM calls, URL validation) is synchronous.

---

# 6. Estimated Cost Per Run

| Parameter | Value |
|-----------|-------|
| GPT-4o-mini input | $0.00015 per 1,000 tokens |
| GPT-4o-mini output | $0.0006 per 1,000 tokens |
| USD → INR | ₹83 per $1 |
| Budget cap | ₹2.00 per run |

**Observed cost (typical run):**

| Metric | Value |
|--------|-------|
| Total tokens consumed | ~7,433 |
| Total cost | ₹0.13 |

**Cost by scenario:**

| Scenario | Estimated Tokens | Estimated Cost |
|----------|-----------------|---------------|
| Clean run — no retries | ~7,400 | ~₹0.13 |
| With 1 retry cycle | ~9,000 | ~₹0.16 |
| With 1 replan cycle | ~10,500 | ~₹0.18 |
| Worst case (retry + replan) | ~13,000 | ~₹0.22 |

---

# 7. Critical Thinking Challenges

## Challenge 1 — The Re-planning Trap

**The risk:** An unanswerable question could loop forever — plan → search → low confidence → replan → repeat, burning tokens indefinitely.

**How our implementation prevents this:**

We enforce hard numeric limits at two independent levels. In `evaluator_constants.py`:

```python
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1
```

The evaluator enforces these inside its decision logic. A retry is only issued if `search_retry_count < MAX_SEARCH_RETRIES`, and a replan is only issued if `replan_count < MAX_REPLANS`. Once both counters are exhausted, the evaluator issues `forced_proceed` — the agent proceeds to synthesis with whatever data it has rather than looping.

The supervisor adds a second layer in `_should_finalize_partial()`:

```python
if state.node_execution_count >= MAX_NODE_EXECUTIONS:  # 12
    return True
if state.failure_counts.get("search_failures", 0) >= MAX_SEARCH_FAILURES:  # 4
    return True
```

This means even if the evaluator keeps producing retry decisions, the supervisor overrides and routes directly to the reporter, setting `is_partial = True`.

**Distinguishing unanswerable vs. needs better strategy:**

The evaluator computes `avg_confidence` against two thresholds: `CONFIDENCE_THRESHOLD = 0.6` and `LOW_CONFIDENCE_THRESHOLD = 0.35`. Confidence between 0.35–0.6 means results exist but are weak — signals "needs a better search strategy" and triggers a retry. Confidence below 0.35 with `all_failed = True` is a stronger signal the query is unanswerable. After `MAX_REPLANS = 1`, the planner uses `planner_replan.txt` with `failure_reason` injected as context to try a structurally different decomposition. If that also fails, `forced_proceed` fires rather than looping again.

---

## Challenge 2 — Citation Truthfulness vs. Citation Coverage

**The problem:** With 12 sources but only 3 high-quality, naively citing all 12 pollutes the reference list. Citing only 3 risks losing important context.

**How our implementation resolves this — 3-stage quality filter:**

**Stage 1 — Searcher (source admission):** Every scraped result is scored by `score_results()` which computes `quality_score = 0.7 × relevance_score + 0.3 × (domain + recency + depth)`. Results below `quality_score >= 0.25` are dropped before they enter the state. Domain scoring gives `.gov`/`.edu`/IEEE/Nature 0.93–0.95, Reddit/Quora 0.5.

**Stage 2 — Synthesiser (context admission):** Before building the LLM context, the synthesiser filters to only `CitationStatus.valid` results, sorts by `quality_score` descending, and takes only `MAX_SYNTHESIS_RESULTS` top entries. Remaining sources are never sent to the LLM.

**Stage 3 — Citation Manager (claim-level verification):** Each claim's `citation_ids` are verified against scraped chunks using `compute_similarity_score`. Only citations with `best_score > SIMILARITY_THRESHOLD (0.4)` survive. Citations below `VERIFICATION_THRESHOLD (0.5)` mark the claim as `verified = False`. Only sources in `used_citation_ids` appear in the final report.

**Direct answer:** Low-quality sources are quarantined progressively — dropped by scoring, excluded from synthesis context, and stripped by the citation manager. A citation only appears in the final report if it passed all three stages. The 12 sources become 3 cited sources naturally without any explicit "cite all vs cite few" threshold.

---

## Challenge 3 — Contradictory Sources

**The problem:** Two sources make directly contradictory factual claims. Three strategies exist: flag both, pick a winner, or refuse to synthesise.

**How our implementation handles it:**

The synthesiser explicitly models conflicts. The LLM is prompted via `synthesiser.txt` to return a `conflicts` array alongside claims, stored directly in the `SynthesisModel`:

```python
conflicts = parsed.get("conflicts", [])
state.synthesis = SynthesisModel(
    claims=claims,
    conflicts=conflicts,
    partial=partial_flag,
)
```

The reporter surfaces them in the report, and `node_logs["SYNTHESIS"]["conflicts"]` is tracked in the debug panel.

**Our chosen strategy — cite both and flag the conflict:** The synthesiser passes both sources to the LLM in the context window and the prompt instructs it to surface disagreements rather than resolve them arbitrarily. The `Claim` model supports multiple `citation_ids`, so a contested claim can cite `[3]` and `[7]` simultaneously with the conflict noted.

**Failure mode analysis:**

| Strategy | Failure Mode |
|---|---|
| Cite both and flag (our approach) | LLM may not reliably detect conflicts if phrased differently or spread across long chunks — no algorithmic conflict detection |
| Pick winner by heuristic (domain/recency score) | A high-domain-score source may be outdated; domain authority ≠ correctness |
| Refuse to synthesise | Leaves the user with no output for contested topics |

**Known gap:** A production improvement would be numerical claim extraction — detecting when two sources state different numbers for the same metric and flagging programmatically, independent of the LLM.

---

## Challenge 5 — The Cost-Quality Curve

**The problem:** More searches and LLM calls improve coverage but increase cost and latency. The supervisor's node execution limit caps both.

**How the limit was tuned:**

`MAX_NODE_EXECUTIONS = 12` in `supervisor_constants.py`. This is intentionally set to allow one full retry cycle (supervisor → planner → searcher → evaluator → supervisor → synthesiser → citation manager → reporter = 7 executions minimum) while preventing runaway loops.

**What happens at the boundary:** When `node_execution_count >= 12`, the supervisor sets `is_partial = True` and routes directly to the reporter. The reporter generates a report from whatever synthesis exists but marks it `partial = True` in metadata. The errors list contains a `loop_limit` entry explaining why — no silent failure.

**The cost guardrail is separate and tighter:** Every node that calls the LLM checks `total_cost > cost_limit` after each call. `cost_limit = 2.0` is a default field on `ResearchState`. This fires `state.abort = True` which the supervisor checks independently of the execution count.

**Could a user configure this as a quality/cost dial?**

Yes — the architecture already supports it. `cost_limit` is a field on `ResearchState` with a default of `2.0`. To expose this as a user-facing dial:

```python
cost_limits = {"Economy": 0.5, "Balanced": 2.0, "Deep Research": 5.0}
state = ResearchState(query=query, cost_limit=cost_limits[mode])
```

The API surface would be three fields: `cost_limit` (float, max spend), `max_depth` (int, maps to `MAX_PLAN_STEPS`), and `quality_mode` (enum: fast / balanced / thorough). The current codebase needs only `ResearchState` to accept these as constructor arguments — no structural changes required.
