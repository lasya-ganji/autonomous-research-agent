from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

THRESHOLD = 0.4
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1


def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    step_evaluations = []
    failed_steps = 0

    for step in state.research_plan:
        step_id = step.step_id
        results = state.search_results.get(step_id, [])

        if not results:
            avg_score = 0.0
            passed = False
            scores = []
        else:
            scores = [r.quality_score for r in results]
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

    # DECISION LOGIC

    if total_steps == 0:
        decision = "replan"
        state.failure_reason = "No steps evaluated"

    elif failed_steps == total_steps:
        if state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "All steps failed due to low confidence"
        else:
            decision = "proceed"
            state.failure_reason = "Max replans reached, proceeding anyway"

    elif failed_steps > 0:
        if state.search_retry_count < MAX_SEARCH_RETRIES:
            decision = "retry"
            state.search_retry_count += 1
            state.failure_reason = "Partial failure, retrying search"
        elif state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "Retry failed, replanning"
        else:
            decision = "proceed"
            state.failure_reason = "Max retries and replans reached"

    else:
        decision = "proceed"
        state.failure_reason = ""

    # Store evaluation
    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    print(f"[EVALUATOR] Decision: {decision}")

    return state