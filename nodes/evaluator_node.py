from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

from services.evaluation.scoring_service import score_results
from services.evaluation.confidence_service import compute_confidence

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time


# UPDATED THRESHOLDS
THRESHOLD = 0.2
LOW_CONF_THRESHOLD = 0.15

MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1


@trace_node("evaluator_node")
def evaluator_node(state: ResearchState) -> ResearchState:
    
    if getattr(state, "skip_eval", False):
        print("⚡ Skipping evaluator (cache hit)")
        return state
    if state.node_execution_count >= 12:
        raise Exception("Max node execution limit reached")

    start_time = time.time()

    input_data = {
        "num_steps": len(state.research_plan),
        "retry_count": state.search_retry_count,
        "replan_count": state.replan_count
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
            confidence = 0.2   
            passed = False
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

        print(f"[EVAL DEBUG] Step {step_id} confidence: {confidence}")
        print(f"[DEBUG] Scored results count for step {step_id}: {len(scored_results) if results else 0}")

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

# EVALUATION DECISION LOGIC

    pass_ratio = (total_steps - failed_steps) / total_steps if total_steps else 0

# BOOST confidence slightly to avoid underflow issues
    avg_confidence = min(1.0, avg_confidence * 1.3)

# DECISION LOGIC

    if total_steps == 0:
        decision = "replan"

    elif pass_ratio >= 0.35:
        decision = "proceed"

    elif avg_confidence >= 0.22:
        decision = "proceed"

    elif state.search_retry_count < MAX_SEARCH_RETRIES:
        decision = "retry"
        state.search_retry_count += 1

    elif state.replan_count < MAX_REPLANS:
        decision = "replan"
        state.replan_count += 1

    else:
        decision = "proceed"

# STORE RESULT
    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
)

    state.overall_confidence = avg_confidence

# DEBUG LOG
    print(f"[EVAL DEBUG] pass_ratio={pass_ratio:.2f}, avg_conf={avg_confidence:.2f}, decision={decision}")

    # DEBUG LOGS
    state.node_logs["evaluator"] = {
        "decision": decision,
        "avg_confidence": avg_confidence,
        "failed_steps": failed_steps,
        "total_steps": total_steps,
        "pass_ratio": pass_ratio,
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

    output_data = {
        "decision": decision,
        "avg_confidence": avg_confidence,
        "failed_steps": failed_steps
    }

    log_node_execution("evaluator", input_data, output_data, start_time)

    state.node_execution_count += 1

    return state