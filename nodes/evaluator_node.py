from models.state import ResearchState
from models.evaluation_models import StepEvaluation, EvaluationResult
from models.error_models import ErrorLog
from models.enums import SeverityEnum, ErrorTypeEnum

from services.evaluation.scoring_service import score_results
from services.evaluation.confidence_service import compute_confidence

from observability.tracing import trace_node
from utils.logger import log_node_execution

import time
from datetime import datetime, timezone


# CONFIG
THRESHOLD = 0.6
LOW_CONF_THRESHOLD = 0.35

MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1

CONFIDENCE_IMPROVEMENT_EPS = 0.02


@trace_node("evaluator_node")
def evaluator_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    try:
        # ensure errors list exists
        if state.errors is None:
            state.errors = []

        if state.node_execution_count >= 12:
            raise Exception("Max node execution limit reached")

        input_data = {
            "num_steps": len(state.research_plan),
            "retry_count": state.search_retry_count,
            "replan_count": state.replan_count,
            "prev_confidence": state.overall_confidence
        }

        step_evaluations = []
        failed_steps = 0
        total_confidence = 0.0

        # STEP EVALUATION LOOP
        for step in state.research_plan:
            step_id = step.step_id
            query = step.question
            results = state.search_results.get(step_id, [])

            confidence = 0.0
            passed = False
            failure_reason = ""

            if not results:
                failure_reason = "no results"

                state.errors.append(
                    ErrorLog(
                        node="evaluator_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"No results for step {step_id}"
                    )
                )

            else:
                # SCORING
                try:
                    scored_results = score_results(results, query, state)
                    state.search_results[step_id] = scored_results
                except Exception as e:
                    state.errors.append(
                        ErrorLog(
                            node="evaluator_node",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            severity=SeverityEnum.ERROR,
                            error_type=ErrorTypeEnum.search_failure,
                            message=f"Scoring failed: {str(e)}"
                        )
                    )
                    scored_results = []

                if not scored_results:
                    failure_reason = "all results filtered"

                else:
                    # CONFIDENCE
                    try:
                        confidence = max(
                            0.0,
                            min(compute_confidence(scored_results, query), 1.0)
                        )
                    except Exception as e:
                        state.errors.append(
                            ErrorLog(
                                node="evaluator_node",
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                severity=SeverityEnum.ERROR,
                                error_type=ErrorTypeEnum.low_confidence,
                                message=f"Confidence computation failed: {str(e)}"
                            )
                        )
                        confidence = 0.0

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

        # AGGREGATION
        total_steps = len(step_evaluations)
        avg_confidence = total_confidence / total_steps if total_steps else 0.0

        prev_conf = state.overall_confidence or 0.0
        improvement = avg_confidence - prev_conf

        no_improvement = abs(improvement) <= CONFIDENCE_IMPROVEMENT_EPS
        low_confidence = avg_confidence < THRESHOLD
        all_failed = failed_steps == total_steps and total_steps > 0

        # DECISION LOGIC
        if total_steps == 0:
            decision = "replan"
            state.failure_reason = "no steps evaluated"

        elif avg_confidence >= THRESHOLD and failed_steps == 0:
            decision = "proceed"
            state.failure_reason = ""

        else:
            if no_improvement:
                decision = "proceed"
                state.failure_reason = "no improvement"

            elif state.search_retry_count < MAX_SEARCH_RETRIES:
                decision = "retry"
                state.search_retry_count += 1

                if all_failed:
                    state.failure_reason = "all steps failed"
                elif low_confidence:
                    state.failure_reason = "low confidence"
                else:
                    state.failure_reason = "partial failure"

            elif state.replan_count < MAX_REPLANS:
                decision = "replan"
                state.replan_count += 1
                state.failure_reason = "retry exhausted"

            else:
                decision = "proceed"
                state.failure_reason = "max retries reached"

        # STORE RESULT
        state.evaluation = EvaluationResult(
            steps=step_evaluations,
            decision=decision
        )

        state.overall_confidence = avg_confidence

        # PROPAGATE QUALITY
        for results in state.search_results.values():
            for r in results:
                cid = getattr(r, "citation_id", None)
                if cid and cid in state.citations:
                    state.citations[cid].quality_score = round(r.quality_score, 3)

        # DEBUG LOGS
        state.node_logs["evaluator"] = {
            "decision": decision,
            "avg_confidence": avg_confidence,
            "failed_steps": failed_steps,
            "total_steps": total_steps
        }

        log_node_execution(
            "evaluator",
            input_data,
            {
                "decision": decision,
                "avg_confidence": avg_confidence
            },
            start_time
        )

        state.node_execution_count += 1

        return state

    # GLOBAL ERROR CATCH
    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="evaluator_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.timeout,
                message=str(e)
            )
        )
        raise e