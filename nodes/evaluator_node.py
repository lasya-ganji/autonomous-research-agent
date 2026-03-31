from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

from services.evaluation.scoring_service import score_results
from services.evaluation.confidence_service import compute_confidence

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time

# CONFIG 

THRESHOLD = 0.6
LOW_CONF_THRESHOLD = 0.4

MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1

CONFIDENCE_IMPROVEMENT_EPS = 0.02  # loop detection


@trace_node("evaluator_node")
def evaluator_node(state: ResearchState) -> ResearchState:

    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    start_time = time.time()

    input_data = {
        "num_steps": len(state.research_plan),
        "retry_count": state.search_retry_count,
        "replan_count": state.replan_count,
        "prev_confidence": state.overall_confidence
    }

    step_evaluations = []
    failed_steps = 0
    total_confidence = 0.0

    # STEP EVALUATION

    for step in state.research_plan:
        step_id = step.step_id
        query = step.question
        results = state.search_results.get(step_id, [])

        confidence = 0.0
        passed = False
        failure_reason = ""

        if not results:
            failure_reason = "no results"

        else:
            scored_results = score_results(results, query)
            state.search_results[step_id] = scored_results

            if not scored_results:
                failure_reason = "all results filtered"

            else:
                confidence = compute_confidence(scored_results)
                passed = confidence >= THRESHOLD

                if not passed:
                    if confidence < LOW_CONF_THRESHOLD:
                        failure_reason = "very low confidence"
                    else:
                        failure_reason = "low confidence"

        total_confidence += confidence

        if not passed:
            failed_steps += 1

        step_evaluations.append(
            StepEvaluation(
                step_id=step_id,
                confidence_score=confidence,
                passed=passed,
                failure_reason=failure_reason
            )
        )

    total_steps = len(step_evaluations)
    avg_confidence = total_confidence / total_steps if total_steps else 0.0

    # LOOP / STAGNATION DETECTION

    prev_conf = state.overall_confidence or 0.0
    no_improvement = abs(avg_confidence - prev_conf) < CONFIDENCE_IMPROVEMENT_EPS

    # DECISION LOGIC (FINAL)

    if total_steps == 0:
        decision = "replan"
        state.failure_reason = "no steps evaluated"

    elif avg_confidence >= THRESHOLD and failed_steps == 0:
        decision = "proceed"
        state.failure_reason = ""

    else:
        # STOP LOOP if no improvement
        if no_improvement:
            decision = "proceed"
            state.failure_reason = "no improvement fallback"

        # Retry first
        elif state.search_retry_count < MAX_SEARCH_RETRIES:
            decision = "retry"
            state.search_retry_count += 1

            if failed_steps == total_steps:
                state.failure_reason = "all steps failed"
            else:
                state.failure_reason = "low confidence"

        # Then replan
        elif state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "retry exhausted"

        # Final fallback (partial output)
        else:
            decision = "proceed"
            state.failure_reason = "max retries reached"

    # STORE RESULTS

    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    state.overall_confidence = avg_confidence

    # DEBUG LOGS (UI)

    state.node_logs["evaluator"] = {
        "decision": decision,
        "avg_confidence": avg_confidence,
        "failed_steps": failed_steps,
        "total_steps": total_steps,
        "no_improvement": no_improvement,
        "steps": [
            {
                "step_id": s.step_id,
                "confidence": s.confidence_score,
                "passed": s.passed,
                "reason": s.failure_reason
            }
            for s in step_evaluations
        ]
    }

    # LOGGER

    output_data = {
        "decision": decision,
        "avg_confidence": avg_confidence,
        "failed_steps": failed_steps
    }

    log_node_execution("evaluator", input_data, output_data, start_time)

    state.node_execution_count += 1

    return state