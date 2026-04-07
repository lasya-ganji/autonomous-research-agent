from datetime import datetime, timezone
from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.logger import log_node_execution
from services.system.cost_tracker import calculate_cost  # ✅ NEW

import time


@trace_node("reporter_node")
def reporter_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    # safety init
    if state.errors is None:
        state.errors = []

    if state.node_logs is None:
        state.node_logs = {}

    try:

        # NO DATA CASE
        if not state.synthesis or not state.synthesis.claims:
            state.report = ReportModel(
                title="Research Report",
                sections=["No reliable data found."],
                citations=[],
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm"
                }
            )
            return state

        # VALID CITATIONS
        valid_citations = {
            cid: c for cid, c in state.citations.items()
            if c.status == CitationStatus.valid
        }

        # COLLECT IDS
        all_claim_ids = set()

        for claim in state.synthesis.claims:
            for cid in claim.citation_ids:
                if cid in valid_citations:
                    all_claim_ids.add(cid)

        sorted_ids = sorted(
            all_claim_ids,
            key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0
        )

        # MAPPING
        id_mapping = {old: f"[{i+1}]" for i, old in enumerate(sorted_ids)}
        state.citation_mapping = id_mapping

        synthesis_lines = []
        used_ids = set()

        for claim in state.synthesis.claims:

            valid_ids = [
                cid for cid in claim.citation_ids
                if cid in valid_citations
            ]

            if not valid_ids:
                continue

            used_ids.update(valid_ids)

            mapped_ids = [id_mapping.get(cid, cid) for cid in valid_ids]

            synthesis_lines.append(
                f"{claim.text} {' '.join(mapped_ids)}"
            )

        # FALLBACK
        if not synthesis_lines:
            state.errors.append(
                ErrorLog(
                    node="reporter_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="No valid synthesis lines after filtering"
                )
            )

            state.report = ReportModel(
                title="Research Report",
                sections=["No valid citations available."],
                citations=[],
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm"
                }
            )
            return state

        synthesis_text = "\n".join(synthesis_lines)

        citations_text = "\n".join(
            [
                f"{id_mapping[cid]} {valid_citations[cid].url}"
                for cid in sorted_ids
            ]
        )

        # LLM CALL
        try:
            prompt = load_prompt("reporter.txt")\
                .replace("{query}", state.query)\
                .replace("{synthesis}", synthesis_text)\
                .replace("{citations}", citations_text)

            res = call_llm(prompt=prompt, temperature=0.2)

            response = res.get("content", "")
            usage = res.get("usage", {})

            # TOKEN TRACKING
            state.total_tokens += usage.get("total_tokens", 0)

            # COST TRACKING
            cost = calculate_cost(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )
            state.total_cost += cost

            # COST GUARDRAIL
            if state.total_cost > state.cost_limit:
                state.abort = True

                state.errors.append(
                    ErrorLog(
                        node="reporter_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.CRITICAL,
                        error_type=ErrorTypeEnum.timeout,
                        message=f"Cost exceeded limit: ₹{state.total_cost}"
                    )
                )

                return state

        except Exception as e:
            state.errors.append(
                ErrorLog(
                    node="reporter_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"LLM call failed: {str(e)}"
                )
            )

            response = "Report generation failed due to LLM error."

        # STORE
        state.used_citation_ids = set(sorted_ids)

        state.report = ReportModel(
            title="Research Report",
            sections=[response],
            citations=sorted_ids,
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm",
                "total_tokens": state.total_tokens,   
                "total_cost": state.total_cost,       
                "errors": [e.model_dump() for e in state.errors]
            }
        )

        state.node_logs["reporter"] = {
            "report_generated": True,
            "citations_used": len(sorted_ids),
            "total_tokens": state.total_tokens,   
            "total_cost": state.total_cost       
        }

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="reporter_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}"
            )
        )

        state.report = ReportModel(
            title="Research Report",
            sections=["Critical failure during report generation."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm"
            }
        )

    log_node_execution("reporter", {}, {}, start_time)

    state.node_execution_count += 1

    return state