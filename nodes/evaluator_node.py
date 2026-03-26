from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

THRESHOLD = 0.5
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1


def score_result(result, query: str) -> float:
    if isinstance(result, dict):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
    else:
        title = getattr(result, "title", "")
        snippet = getattr(result, "snippet", "")

    text = (title + " " + snippet).lower()
    query_terms = query.lower().split()

    if not query_terms:
        return 0.0

    match_count = sum(1 for term in query_terms if term in text)
    return match_count / len(query_terms)


def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    # execution safety
    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    step_evaluations = []
    failed_steps = 0

    for step in state.research_plan:
        step_id = step.step_id
        query = step.question
        results = state.search_results.get(step_id, [])

        if not results:
            scores = []
            avg_score = 0.0
            passed = False
        else:
            scores = [score_result(r, query) for r in results]
            avg_score = sum(scores) / len(scores)
            passed = avg_score >= THRESHOLD

        print(f"[EVALUATOR DEBUG] Step {step_id} scores:", scores, "avg:", avg_score)

        if not passed:
            failed_steps += 1

        step_evaluations.append(
            StepEvaluation(
                step_id=step_id,
                confidence_score=avg_score,
                passed=passed
            )
        )

    total_steps = len(step_evaluations)

    print(f"[EVALUATOR] Total steps: {total_steps}, Failed: {failed_steps}")

    # decision logic (PRD aligned)
    if total_steps == 0:
        decision = "replan"
        state.failure_reason = "No steps evaluated"

    elif failed_steps == total_steps:
        if state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "All steps failed"
        else:
            decision = "proceed"
            state.failure_reason = "Max replans reached"

    elif failed_steps > 0:
        if state.search_retry_count < MAX_SEARCH_RETRIES:
            decision = "retry"
            state.search_retry_count += 1
            state.failure_reason = "Partial failure"
        elif state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "Retry failed"
        else:
            decision = "proceed"
            state.failure_reason = "Max retries reached"

    else:
        decision = "proceed"
        state.failure_reason = ""

    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    print(f"[EVALUATOR] Decision: {decision}")

    state.node_execution_count += 1

    return state