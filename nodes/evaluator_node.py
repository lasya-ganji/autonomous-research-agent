import time
from datetime import datetime, timezone
from typing import Dict, List

from config.constants.node_names import NodeNames
from models.evaluation_models import EvaluationResult, StepEvaluation
from models.enums import ErrorTypeEnum, SeverityEnum
from models.error_models import ErrorLog
from models.state import ResearchState
from observability.tracing import trace_node
from services.evaluation.confidence_service import compute_confidence
from services.evaluation.scoring_service import score_results
from utils.logger import log_node_execution


# CONFIG
THRESHOLD = 0.6
LOW_CONF_THRESHOLD = 0.35
MAX_SEARCH_RETRIES = 1
MAX_REPLANS = 1
CONFIDENCE_IMPROVEMENT_EPS = 0.02


@trace_node(NodeNames.EVALUATOR)
def evaluator_node(state: ResearchState) -> ResearchState:

    # Safety init
    if state.errors is None:
        state.errors = []
    if state.node_logs is None:
        state.node_logs = {}
    if not hasattr(state, "failure_counts") or state.failure_counts is None:
        state.failure_counts = {
            "search_failures": 0,
            "parsing_failures": 0,
            "low_confidence": 0,
        }


    input_data: Dict = {
        "num_steps": len(state.research_plan),
        "retry_count": state.search_retry_count,
        "replan_count": state.replan_count,
        "prev_confidence": state.overall_confidence,
    }

    try:
        step_evaluations: List[StepEvaluation] = []
        failed_steps = 0
        total_confidence = 0.0
        step_debug = {}

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

            # -------------------------------
            # NO RESULTS
            # -------------------------------
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
                # -------------------------------
                # SCORING (IMPORTANT: pass state)
                # -------------------------------
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
                            message=f"Scoring failed: {str(e)}",
                        )
                    )
                    scored_results = []

                # -------------------------------
                # ALL FILTERED
                # -------------------------------
                if not scored_results:
                    failure_reason = "all results filtered"
                    state.failure_counts["low_confidence"] += 1

                else:
                    # -------------------------------
                    # CONFIDENCE
                    # -------------------------------
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
                                message=f"Confidence failed: {str(e)}",
                            )
                        )
                        confidence = 0.0

                    passed = confidence >= THRESHOLD

                    if not passed:
                        state.failure_counts["low_confidence"] += 1
                        if confidence < LOW_CONF_THRESHOLD:
                            failure_reason = "very low confidence"
                        else:
                            failure_reason = "low confidence"

            total_confidence += confidence
            if not passed:
                failed_steps += 1

            step_debug[step_id] = {
                "confidence": confidence,
                "passed": passed,
                "reason": failure_reason
            }

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
        no_improvement = abs(improvement) <= CONFIDENCE_IMPROVEMENT_EPS

        low_confidence = avg_confidence < THRESHOLD
        all_failed = failed_steps == total_steps and total_steps > 0

        # -------------------------------
        # DECISION LOGIC (FINAL CORRECT)
        # -------------------------------
        if total_steps == 0:
            decision = "replan"
            state.failure_reason = "no steps evaluated"

        elif avg_confidence >= THRESHOLD and failed_steps == 0:
            decision = "proceed"
            state.failure_reason = ""

        else:
            # GLOBAL retry first
            if state.search_retry_count < MAX_SEARCH_RETRIES:
                decision = "retry"
                state.search_retry_count += 1

                if all_failed:
                    state.failure_reason = "all steps failed"
                elif low_confidence:
                    state.failure_reason = "low confidence"
                else:
                    state.failure_reason = "partial failure"

            # THEN replan
            elif state.replan_count < MAX_REPLANS:
                decision = "replan"
                state.replan_count += 1
                state.failure_reason = "retry exhausted"

            # FINAL fallback
            else:
                decision = "proceed"
                state.failure_reason = "max retries and replans exhausted"

        # -------------------------------
        # STATE UPDATE
        # -------------------------------
        state.evaluation = EvaluationResult(
            steps=step_evaluations,
            decision=decision
        )
        state.overall_confidence = avg_confidence

        # Update citation scores
        for results in state.search_results.values():
            for r in results:
                cid = getattr(r, "citation_id", None)
                if cid and cid in state.citations:
                    state.citations[cid].quality_score = round(r.quality_score, 3)

        # -------------------------------
        # OBSERVABILITY (IMPORTANT)
        # -------------------------------
        node_name = NodeNames.EVALUATOR

        existing_log = state.node_logs.get(node_name, {})

        existing_log.update({
            "decision": decision,
            "avg_confidence": avg_confidence,
            "failed_steps": failed_steps,
            "total_steps": total_steps,
            "no_improvement": no_improvement,
            "improvement": round(improvement, 4),
            "step_debug": step_debug,
            "failure_counts": state.failure_counts
        })

        state.node_logs[node_name] = existing_log

        log_node_execution(
            "evaluator_node",
            input_data,
            {"decision": decision, "avg_confidence": avg_confidence}
        )

        state.node_execution_count += 1
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

        state.evaluation = EvaluationResult(steps=[], decision="proceed")
        state.overall_confidence = state.overall_confidence or 0.0

        node_name = NodeNames.EVALUATOR

        existing_log = state.node_logs.get(node_name, {})

        existing_log.update({
            "decision": "proceed",
            "avg_confidence": state.overall_confidence,
            "error": str(e)
        })

        state.node_logs[node_name] = existing_log

        log_node_execution(
            "evaluator_node",
            input_data,
            {"decision": "proceed"}
        )

        state.node_execution_count += 1
        return state