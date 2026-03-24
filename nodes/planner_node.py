from typing import List
from pydantic import ValidationError
from datetime import datetime, timezone

from models.state import ResearchState
from models.planner_models import PlanStep
from models.error_models import ErrorLog
from tools.llm_tool import call_llm


def planner_node(state: ResearchState) -> ResearchState:
    
    # Extract input
    query: str = state.query

    if not query:
        raise ValueError("Query missing in state")

    # Replan logic
    is_replan: bool = state.replan_count > 0

    # TEMPORARY PROMPT (prompt files not implemented yet)
    if is_replan:
        prompt = f"""
You are a research planner.

The previous plan was not sufficient. Improve it.

Query: {query}

Generate 3-5 better sub-questions.

Return ONLY a JSON array in this format:
[
  {{
    "step_id": 1,
    "question": "...",
    "priority": 5
  }}
]

Rules:
- Do NOT return empty list
- No explanation
"""
    else:
        prompt = f"""
You are a research planner.

Break the following query into 3-5 meaningful sub-questions.

Query: {query}

Return ONLY a JSON array in this format:
[
  {{
    "step_id": 1,
    "question": "...",
    "priority": 5
  }}
]

Rules:
- Generate at least 3 steps
- Do NOT return empty list
- No explanation
"""

    # Call LLM
    response = call_llm(
        prompt=prompt,
        temperature=0.2,
        expect_json=True
    )

    print("\n[PLANNER RESPONSE]:", response)

    # Validate response
    plan: List[PlanStep] = []

    if isinstance(response, list):
        for item in response:
            try:
                plan.append(PlanStep(**item))
            except ValidationError as e:
                state.errors.append(
                    ErrorLog(
                        node="planner_node",
                        error_type="parsing_error",
                        message=str(e),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity="ERROR"
                    )
                )
    else:
        state.errors.append(
            ErrorLog(
                node="planner_node",
                error_type="parsing_error",
                message=str(response),
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity="ERROR"
            )
        )

    # Fallback plan (safety)
    if not plan:
        plan = [
            PlanStep(
                step_id=1,
                question=query,
                priority=5
            )
        ]

    # Sort by priority
    plan = sorted(plan, key=lambda x: x.priority, reverse=True)

    # Store plan
    state.research_plan = plan

    # Reset search results
    state.search_results = {}

    # Track unresolved steps
    state.unresolved_steps = list(range(len(plan)))

    # Reset retry counter
    state.search_retry_count = 0

    # Observability
    state.node_execution_count += 1

    return state