from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult


# Base threshold for confidence (adjusted for current scoring system)
THRESHOLD = 0.45

# Retry limits (as per design decisions)
MAX_SEARCH_RETRIES = 1   # Only 1 retry for low-confidence results
MAX_REPLANS = 1          # Only 1 replan allowed


def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    step_evaluations = []
    failed_steps = 0

    #  Iterate through results of each planned step
    for step_id, results in state.search_results.items():

        # Extract quality scores (already computed in scoring_service)
        scores = [r.quality_score for r in results]

        # Compute average confidence for this step
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Check if this step passed quality threshold
        passed = avg_score >= THRESHOLD

        if not passed:
            failed_steps += 1

        # Store per-step evaluation
        step_evaluations.append(
            StepEvaluation(
                step_id=step_id,
                confidence_score=avg_score,
                passed=passed
            )
        )

    total_steps = len(step_evaluations)

    # DECISION LOGIC (CORE INTELLIGENCE)

    # Case 0: No steps evaluated → force replan
    if total_steps == 0:
        decision = "replan"

    # Case 1: Majority steps failed → directly replan
    elif failed_steps / total_steps > 0.5:

        if state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
        else:
            # Prevent infinite loop → proceed anyway
            decision = "proceed"

    # Case 2: Partial failure → retry once, then replan
    elif failed_steps > 0:

        # First try → retry search (same plan)
        if state.search_retry_count < MAX_SEARCH_RETRIES:
            decision = "retry"
            state.search_retry_count += 1

        # If retry already done → replan
        elif state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1

        # If both retry + replan done → proceed
        else:
            decision = "proceed"

    # Case 3: All steps passed → proceed
    else:
        decision = "proceed"

    #  Store evaluation result in state
    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    print(f"[EVALUATOR] Decision: {decision}")

    return state