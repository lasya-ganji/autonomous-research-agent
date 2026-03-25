from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

THRESHOLD = 0.6


def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    step_evaluations = []
    failed_steps = 0

    # Loop through search results per step
    for step_id, results in state.search_results.items():

        scores = []

        for r in results:
            # Use precomputed score from scoring_service
            scores.append(r.quality_score)

        # Compute average score per step
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0.0

        # Threshold check
        passed = avg_score >= THRESHOLD

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

    # Decision logic
    if total_steps == 0:
        decision = "replan"

    elif failed_steps / total_steps > 0.5:
        decision = "replan"
        state.replan_count += 1

    elif failed_steps > 0:
        decision = "retry"
        state.search_retry_count += 1

    else:
        decision = "proceed"

    # Store evaluation result in state
    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    print(f"[EVALUATOR] Decision: {decision}")

    return state