import time
from datetime import datetime, timezone
from typing import Dict, List

from config.constants.node_constants.node_names import NodeNames
from config.constants.node_constants.evaluator_constants import (
    CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_SEARCH_RETRIES,
    MAX_REPLANS
)

from models import state
from models.evaluation_models import EvaluationResult, StepEvaluation
from models.enums import ErrorTypeEnum, SeverityEnum
from models.error_models import ErrorLog
from models.state import ResearchState

from observability.tracing import trace_node

from services.evaluation.confidence_service import compute_confidence
from services.evaluation.scoring_service import score_results

from utils.logger import log_node_execution


@trace_node(NodeNames.EVALUATOR)
def evaluator_node(state: ResearchState) -> ResearchState:

    # -------------------------------
    # SAFETY INIT
    # -------------------------------
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}

    # IMPORTANT: reset only per-run counters
    state.failure_counts["search_failures"] = 0
    state.failure_counts["low_confidence"] = 0

    input_data: Dict = {
        "num_steps": len(state.research_plan),
        "retry_count": state.search_retry_count,
        "replan_count": state.replan_count,
        "prev_confidence": state.overall_confidence,
    }

    try:
        
        start_tokens = state.total_tokens
        start_cost = state.total_cost
        
        step_evaluations: List[StepEvaluation] = []
        failed_steps = 0
        total_confidence = 0.0

        # -------------------------------
        # PER-STEP EVALUATION
        # -------------------------------
        for step in state.research_plan:
            step_id = step.step_id
            query = step.question
            results = state.search_results.get(step_id, [])

            confidence = 0.0
            passed = False
            failure_reason = ""

            if not results:
                failure_reason = "no results"
                state.failure_counts["search_failures"] += 1

                state.errors.append(
                    ErrorLog(
                        node="evaluator_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.search_failure,
                        message=f"No results for step {step_id}",
                    )
                )

            else:
                try:
                    scored_results = score_results(results, query, state)
                    state.search_results[step_id] = scored_results
                except Exception as e:
                    state.errors.append(
                        ErrorLog(
                            node="evaluator_node",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            severity=SeverityEnum.ERROR,
                            error_type=ErrorTypeEnum.system_error,
                            message=f"Scoring failed (step {step_id}): {str(e)}",
                        )
                    )
                    scored_results = []

                if not scored_results:
                    failure_reason = "all results filtered"
                    state.failure_counts["low_confidence"] += 1

                else:
                    try:
                        confidence = compute_confidence(scored_results, query)
                        confidence = max(0.0, min(confidence, 1.0))
                    except Exception as e:
                        state.errors.append(
                            ErrorLog(
                                node="evaluator_node",
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                severity=SeverityEnum.ERROR,
                                error_type=ErrorTypeEnum.system_error,
                                message=f"Confidence failed (step {step_id}): {str(e)}",
                            )
                        )
                        confidence = 0.0

                    passed = confidence >= CONFIDENCE_THRESHOLD

                    if not passed:
                        state.failure_counts["low_confidence"] += 1
                        failure_reason = (
                            "very low confidence"
                            if confidence < LOW_CONFIDENCE_THRESHOLD
                            else "low confidence"
                        )

            total_confidence += confidence

            if not passed:
                failed_steps += 1

            step_evaluations.append(
                StepEvaluation(
                    step_id=step_id,
                    confidence_score=confidence,
                    passed=passed,
                    failure_reason=failure_reason,
                )
            )

        # -------------------------------
        # AGGREGATION
        # -------------------------------
        total_steps = len(step_evaluations)
        avg_confidence = total_confidence / total_steps if total_steps else 0.0

        prev_conf = state.overall_confidence or 0.0
        improvement = avg_confidence - prev_conf
        no_improvement = improvement <= 0

        low_confidence = avg_confidence < CONFIDENCE_THRESHOLD

        all_failed = failed_steps == total_steps and total_steps > 0

        # -------------------------------
        # DECISION LOGIC
        # -------------------------------
        if total_steps == 0:
            decision = "replan"
            state.failure_reason = "no steps evaluated"

        elif avg_confidence >= CONFIDENCE_THRESHOLD:
            decision = "proceed"
            state.failure_reason = ""

        else:
            if state.search_retry_count < MAX_SEARCH_RETRIES and not no_improvement:
                decision = "retry"
                state.search_retry_count += 1
                state.failure_reason = (
                    "all steps failed" if all_failed else "low confidence"
                )

            elif state.replan_count < MAX_REPLANS:
                decision = "replan"
                state.replan_count += 1
                state.failure_reason = "retry exhausted"

            else:
                decision = "forced_proceed"
                state.failure_reason = "max retries and replans exhausted"

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        state.evaluation = EvaluationResult(
            steps=step_evaluations,
            decision=decision
        )
        state.overall_confidence = avg_confidence

        # -------------------------------
        # UPDATE CITATION SCORES
        # -------------------------------
        for results in state.search_results.values():
            for r in results:
                cid = getattr(r, "citation_id", None)
                if cid and cid in state.citations:
                    state.citations[cid].quality_score = round(r.quality_score, 3)

        # -------------------------------
        # LOGGING
        # -------------------------------
        node_tokens = state.total_tokens - start_tokens
        node_cost = state.total_cost - start_cost
        
        node_name = NodeNames.EVALUATOR
        existing_log = state.node_logs.get(node_name, {})

        existing_log.update({
            "decision": decision,
            "avg_confidence": round(avg_confidence, 4),
            "failed_steps": failed_steps,
            "total_steps": total_steps,
            "improvement": round(improvement, 4),
            "no_improvement": no_improvement,
            "low_confidence": low_confidence,
            "all_failed": all_failed,
            "failure_counts": dict(state.failure_counts),
            "errors_count": len(state.errors),
            "node_tokens": node_tokens,
            "node_cost": round(node_cost, 4),
            "total_cost": round(state.total_cost, 4)
        })

        state.node_logs[node_name] = existing_log

        log_node_execution(
            "evaluator_node",
            input_data,
            {
                "decision": decision,
                "avg_confidence": round(avg_confidence, 4),
            }
        )

        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="evaluator_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=str(e),
            )
        )

        # safe fallback
        state.evaluation = EvaluationResult(steps=[], decision="forced_proceed")
        state.overall_confidence = state.overall_confidence or 0.0

        state.node_logs[NodeNames.EVALUATOR] = {
            "decision": "forced_proceed",
            "error": str(e),
            "errors_count": len(state.errors),
        }

        log_node_execution(
            "evaluator_node",
            input_data,
            {"decision": "forced_proceed"}
        )

        return state