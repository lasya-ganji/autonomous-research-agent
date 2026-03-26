from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

from services.evaluation.scoring_service import score_results
from services.evaluation.confidence_service import compute_confidence

from observability.tracing import trace_node

THRESHOLD = 0.5  
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1


@trace_node("evaluator_node")
def evaluator_node(state: ResearchState) -> ResearchState:
    print("Evaluator Node")

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    step_evaluations = []
    failed_steps = 0

    for step in state.research_plan:
        step_id = step.step_id
        query = step.question
        results = state.search_results.get(step_id, [])

        if not results:
            confidence = 0.0
            passed = False
            scores = []
        else:
            # 1. scoring
            scored_results = score_results(results, query)

            # store back
            state.search_results[step_id] = scored_results

            # extract scores
            scores = [r.quality_score for r in scored_results]

            # 2. confidence 
            confidence = compute_confidence(scored_results)

            # 3. decision based on confidence
            passed = confidence >= THRESHOLD

        print(f"[EVALUATOR DEBUG] Step {step_id} scores:", scores)
        print(f"[EVALUATOR DEBUG] Step {step_id} confidence:", confidence)
        print(f"[DEBUG] Step {step_id} passed: {passed}")

        if not passed:
            failed_steps += 1

        step_evaluations.append(
            StepEvaluation(
                step_id=step_id,
                confidence_score=confidence,
                passed=passed
            )
        )

    total_steps = len(step_evaluations)

    print(f"[EVALUATOR] Total steps: {total_steps}, Failed: {failed_steps}")

    # Decision logic
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
            state.failure_reason = "All steps failed after max replans"
            print("[WARNING] Proceeding with low confidence")

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