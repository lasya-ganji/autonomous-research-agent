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

**No persistent state or session memory:**
Each agent run starts with a fresh `ResearchState`. There is no caching of prior research, no cross-session memory, and no deduplication against previously researched topics. Every invocation is fully stateless.

**Domain authority uses rule-based heuristics, not commercial SEO metrics:**
We estimate source credibility using simple, rule-based heuristics instead of paid SEO metrics. This includes tier-based scoring (e.g., high-authority, research, government/education, blogs), TLD boosts like .gov or .edu, and configurable rules in domain_authority.json.

This approach keeps things fast, predictable, and cost-efficient since it avoids third-party APIs. The downside is that it doesn’t capture richer signals like backlinks or real-time authority scores from tools like `Moz` .

In the future, we could enhance this by integrating external APIs (if budget allows), moving toward more data-driven credibility scoring while still keeping a lightweight default option.

**Embedding model trade-off (accuracy vs latency):**
For semantic tasks (like similarity checks and coverage signals), we use `all-MiniLM-L6-v2` because it’s lightweight and fast, making it suitable for real-time use.

Stronger models like `BAAI/bge-large-en`could improve accuracy, especially for subtle meaning differences, but they come with higher compute cost and slower response times.

So this is a conscious trade-off: we prioritize speed and efficiency over maximum accuracy. A good next step would be a hybrid approach—using a fast model for initial filtering and a larger model for final scoring, depending on complexity or performance needs.

**HTML-only content retrieval (format limitation):**
The system filters out binary files (PDF, DOCX, PPT) and non-article URLs, focusing only on HTML pages. This keeps the pipeline simple, fast, and reliable.

The trade-off is reduced coverage of high-quality sources like `research papers` and  `technical documents `, which may lead to missing authoritative information. Future improvements could include adding document parsing to support multi-format content.


## Production Gaps

**No LLM fallback:** No fallback to an alternative provider if OpenAI is unavailable.

**No persistent storage:** Results are not saved — refreshing the UI loses all outputs.

**No async execution:** All I/O (scraping, LLM calls, URL validation) is synchronous.

**Cold-start latency:** `embedding_service.py` loads `all-MiniLM-L6-v2` at import, adding 2–5s startup delay.

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

This was the first thing that worried us when we designed the evaluator. If a query has no good
sources on the web, the agent could just keep replanning forever — spending tokens and getting
nowhere. We needed a way to stop that without silently giving up on queries that just needed a
better search strategy.

We ended up with two separate hard limits. In `evaluator_constants.py`:

```python
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1
```

The evaluator checks these before issuing any retry or replan decision. If both counters are
exhausted, it issues `forced_proceed` — meaning the agent moves forward to synthesis with
whatever it has, rather than looping again.

But we also added a second safety layer in the supervisor via `_should_finalize_partial()`:

```python
if state.node_execution_count >= MAX_NODE_EXECUTIONS:  # 12
    return True
if state.failure_counts.get("search_failures", 0) >= MAX_SEARCH_FAILURES:  # 4
    return True
```

The reason we have both is that the evaluator's counters only track deliberate retries and
replans, but the execution count catches anything unexpected — like a node crashing and being
re-entered in a way the counters don't see.

For distinguishing "unanswerable" from "needs a better approach" — we use two confidence
thresholds: `CONFIDENCE_THRESHOLD = 0.6` and `LOW_CONFIDENCE_THRESHOLD = 0.35`.
Confidence between 0.35–0.6 means the searcher found something but it was weak, which
suggests the search strategy was wrong, not that the topic is unanswerable. That triggers a
retry. Below 0.35 with `all_failed = True` is a much stronger signal that there's genuinely
nothing useful out there. In that case the replanner gets `failure_reason` injected as context
via `planner_replan.txt` to try a structurally different decomposition — but only once. After
that, we give up and deliver whatever partial output exists. It's not a perfect heuristic but it
works well enough for the queries we tested.

---

## Challenge 2 — Citation Truthfulness vs. Citation Coverage

Honestly, this one we didn't fully solve in one shot. Our first instinct was to just send all
sources to the synthesiser and let the LLM pick the good ones — but that produced reports that
cited Reddit threads and marketing pages alongside actual research. So we built a progressive
filtering system with three stages.

**Stage 1 — the searcher decides which sources even enter state.** Every result gets a
`quality_score` computed as `0.7 × relevance_score + 0.3 × (domain + recency + depth)`.
Anything below 0.25 is dropped before it's ever stored. Domain scoring is hardcoded — `.gov`,
`.edu`, IEEE, Nature get 0.93–0.95; Reddit and Quora get 0.5. This was a pragmatic choice —
we knew these domains behaved consistently enough to justify hard scores.

**Stage 2 — the synthesiser decides which sources the LLM even sees.** Before building the
context window, it filters to `CitationStatus.valid` sources only, sorts by `quality_score`
descending, and caps at `MAX_SYNTHESIS_RESULTS`. Sources that didn't make the cut never
reach the LLM at all, so it can't cite them even if it wanted to.

**Stage 3 — the citation manager decides which citations survive per claim.** Each cited
source is checked against the actual scraped text using embedding similarity. If the best
chunk score for a source is ≤ 0.4, that citation is marked as hallucinated and stripped from
the claim. If no citation survives, the claim is marked `verified = False`.

The result is that low-quality sources get dropped progressively rather than at one big
decision point. In practice, we found this brought citation counts from 8–10 down to 3–5 in
most runs, which felt right — those were genuinely the sources the report was grounded in.

The trade-off we're less happy about is Stage 2. Capping at `MAX_SYNTHESIS_RESULTS` means
that sometimes a relevant source ranked 13th gets cut. We don't have a great answer for
that — in a production version we'd probably want the synthesiser to use more sources for
context even if they don't all end up cited.

---

## Challenge 3 — Contradictory Sources

We hit this concretely during testing — a query about online learning completion rates
returned two sources citing different figures for the same statistic. We had three options:
pick one, cite both and flag it, or refuse to synthesise. Refusing to synthesise felt wrong —
that would make the agent useless for any contested topic. Picking one based on domain score
felt dangerous — a high-ranked source can still be wrong or outdated.

So we went with flagging. The synthesiser prompt (`synthesiser.txt`) explicitly asks the LLM to
surface disagreements rather than resolve them, and the output schema has a `conflicts` array
alongside `claims`:

```python
conflicts = parsed.get("conflicts", [])
state.synthesis = SynthesisModel(
    claims=claims,
    conflicts=conflicts,
    partial=partial_flag,
)
```

A claim can carry multiple `citation_ids` so a contested statement can reference both `[3]`
and `[7]` simultaneously. The reporter surfaces the conflict in the final output, and the debug
panel shows `node_logs["SYNTHESIS"]["conflicts"]` for inspection.

The failure mode we know exists: the LLM doesn't reliably detect conflicts when the
contradictory statements are phrased differently or buried deep in long chunks. There's no
algorithmic check — it's entirely LLM-dependent. A production version would do numerical
claim extraction first, flag any cases where two sources give different numbers for the same
metric, and only then pass those to the LLM for resolution. We didn't have time to build that.

| Strategy | Failure Mode |
|---|---|
| Cite both and flag (our approach) | LLM may not reliably detect conflicts if phrased differently or spread across long chunks |
| Pick winner by heuristic (domain/recency score) | A high-domain-score source may be outdated; domain authority ≠ correctness |
| Refuse to synthesise | Leaves the user with no output for contested topics |

---

## Challenge 5 — The Cost-Quality Curve

We picked `MAX_NODE_EXECUTIONS = 12` fairly pragmatically. The minimum viable path through
the graph is 7 executions (supervisor → planner → searcher → evaluator → synthesiser →
citation manager → reporter), which means 12 gives us exactly one retry/replan cycle with a
small buffer. We didn't do extensive tuning — we chose 12, ran a few queries, and it felt like
it never hit the limit on normal queries but did catch our infinite-loop test cases.

What happens when the limit is hit: the supervisor sets `is_partial = True` and routes
directly to the reporter. The reporter still generates a report from whatever synthesis exists,
but marks it `partial = True` in the output metadata. We made sure the error is explicit —
the errors list gets a `loop_limit` entry so anyone reading the output knows why it's
incomplete.

There's a separate cost guardrail that's tighter than the execution limit. Every LLM-calling
node checks `total_cost > cost_limit` after each call. If it's exceeded, `abort = True` fires
and the supervisor routes to the reporter immediately, regardless of execution count. The
default `cost_limit = 2.0` (INR) — in practice we've never seen a run exceed ₹0.22 even in
worst-case retry+replan scenarios, so the limit is generous. But it's there for safety.

On whether this could be a user-facing dial — yes, and the architecture already supports it.
`cost_limit` is just a field on `ResearchState` with a default value. You could expose three
presets:

```python
cost_limits = {"Economy": 0.5, "Balanced": 2.0, "Deep Research": 5.0}
state = ResearchState(query=query, cost_limit=cost_limits[mode])
```

The honest answer is we didn't prioritise building that UI — the Streamlit app doesn't expose
it. It would need a slider in the sidebar and that's about it on the frontend side. The backend
already handles it correctly.
