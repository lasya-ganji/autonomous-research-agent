from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult

from services.evaluation.scoring_service import score_results
from services.evaluation.confidence_service import compute_confidence

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time

# CONFIG

THRESHOLD = 0.6
LOW_CONF_THRESHOLD = 0.35

MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1

CONFIDENCE_IMPROVEMENT_EPS = 0.02


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

    # evaluate each step independently
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
            # scoring 
            scored_results = score_results(results, query)
            state.search_results[step_id] = scored_results
            
            for r in scored_results:
                print({
                    "title": r.title[:50],
                    "relevance": r.relevance_score,
                    "domain": r.domain_score,
                    "recency": r.recency_score,
                    "depth": r.depth_score,
                    "quality": r.quality_score
                })

            if not scored_results:
                failure_reason = "all results filtered"

            else:
                # confidence (embedding-based)
                confidence = max(0.0, min(compute_confidence(scored_results,query), 1.0))

                # quality guard: ensure top result is meaningful
                top_quality = scored_results[0].quality_score

                if top_quality < 0.35:
                    confidence = 0.0
                    passed = False
                    failure_reason = "low quality results"
                else:
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

    # loop / stagnation detection
    prev_conf = state.overall_confidence or 0.0
    improvement = avg_confidence - prev_conf

    # use absolute change to avoid small fluctuations from embeddings
    no_improvement = abs(improvement) <= CONFIDENCE_IMPROVEMENT_EPS

    low_confidence = avg_confidence < THRESHOLD
    all_failed = failed_steps == total_steps and total_steps > 0

    # decision logic
    if total_steps == 0:
        decision = "replan"
        state.failure_reason = "no steps evaluated"

    elif avg_confidence >= THRESHOLD and failed_steps == 0:
        decision = "proceed"
        state.failure_reason = ""

    else:
        # stop loop if no meaningful improvement
        if no_improvement:
            decision = "proceed"
            state.failure_reason = "no improvement"

        # retry search first
        elif state.search_retry_count < MAX_SEARCH_RETRIES:
            decision = "retry"
            state.search_retry_count += 1

            if all_failed:
                state.failure_reason = "all steps failed"
            elif low_confidence:
                state.failure_reason = "low confidence"
            else:
                state.failure_reason = "partial failure"

        # then replan
        elif state.replan_count < MAX_REPLANS:
            decision = "replan"
            state.replan_count += 1
            state.failure_reason = "retry exhausted"

        # fallback to partial output
        else:
            decision = "proceed"
            state.failure_reason = "max retries reached"

    # store evaluation result
    state.evaluation = EvaluationResult(
        steps=step_evaluations,
        decision=decision
    )

    state.overall_confidence = avg_confidence

    # propagate quality scores to citations
    for results in state.search_results.values():
        for r in results:
            cid = getattr(r, "citation_id", None)
            if cid and cid in state.citations:
                state.citations[cid].quality_score = round(r.quality_score, 3)

    # debug logs
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

    # structured logging
    output_data = {
        "decision": decision,
        "avg_confidence": avg_confidence,
        "failed_steps": failed_steps
    }

    log_node_execution("evaluator", input_data, output_data, start_time)

    state.node_execution_count += 1

    return state