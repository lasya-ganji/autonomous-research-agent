import time
from datetime import datetime, timezone
from typing import Dict, List, Set

from observability.tracing import trace_node

from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames
from services.system.cost_tracker import calculate_cost


@trace_node(NodeNames.REPORTER)
def reporter_node(state: ResearchState) -> ResearchState:
    start_time = time.time()

    try:
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

        # NO DATA CASE
        if not state.synthesis or not state.synthesis.claims:
            state.report = ReportModel(
                title="Research Report",
                sections=["No reliable data found."],
                citations=[],
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm",
                    "partial": True,
                    "errors": [e.model_dump() for e in state.errors],
                },
            )
            state.node_execution_count += 1
            state.node_logs[NodeNames.REPORTER] = {
                "report_generated": False,
                "citations_used": 0,
                "conflicts": 0,
                "total_tokens": state.total_tokens,
                "total_cost": state.total_cost,
            }
            log_node_execution("reporter", {}, {}, start_time)
            return state

        valid_citations: Dict[str, object] = {
            cid: c for cid, c in state.citations.items() if c.status == CitationStatus.valid
        }

        # COLLECT IDs referenced by claims
        all_claim_ids: Set[str] = set()
        for claim in state.synthesis.claims:
            for cid in getattr(claim, "citation_ids", []):
                if cid in valid_citations:
                    all_claim_ids.add(cid)

        sorted_ids = sorted(
            all_claim_ids,
            key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0,
        )

        # Mapping old numeric IDs to new sequential IDs
        id_mapping = {old: f"[{i + 1}]" for i, old in enumerate(sorted_ids)}
        state.citation_mapping = id_mapping

        synthesis_lines: List[str] = []
        used_ids: Set[str] = set()

        for claim in state.synthesis.claims:
            valid_ids = [cid for cid in getattr(claim, "citation_ids", []) if cid in valid_citations]

            if not valid_ids:
                synthesis_lines.append(f"{claim.text} [UNVERIFIED]")
                continue

            used_ids.update(valid_ids)
            mapped_ids = [id_mapping.get(cid, cid) for cid in valid_ids]
            synthesis_lines.append(
                f"{claim.text} {' '.join(mapped_ids)} "
                f"(confidence: {round(getattr(claim, 'citation_confidence', 0.0), 2)}, verified: {getattr(claim, 'verified', False)})"
            )

        if not synthesis_lines:
            state.report = ReportModel(
                title="Research Report",
                sections=["No valid citations available."],
                citations=[],
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm",
                    "partial": True,
                    "errors": [e.model_dump() for e in state.errors],
                },
            )
            return state

        synthesis_text = "\n".join(synthesis_lines)

        # FULL CITATION METADATA
        citations_list: List[Dict] = []
        citations_text_lines: List[str] = []
        for cid in sorted_ids:
            c = valid_citations[cid]
            entry = {
                "id": id_mapping[cid],
                "url": c.url,
                "title": getattr(c, "title", ""),
                "quality_score": getattr(c, "quality_score", None),
                "accessed_at": datetime.now().isoformat(),
            }
            citations_list.append(entry)
            citations_text_lines.append(f"{entry['id']} {entry['title']} - {entry['url']}")

        citations_text = "\n".join(citations_text_lines)

        # CONFLICTS SECTION (if any)
        conflicts_text = ""
        if state.synthesis.conflicts:
            conflicts_text = "\n\nConflicts:\n" + "\n".join(state.synthesis.conflicts)

        prompt = (
            load_prompt("reporter.txt")
            .replace("{query}", state.query)
            .replace("{synthesis}", synthesis_text + conflicts_text)
            .replace("{citations}", citations_text)
        )

        res = call_llm(prompt=prompt, temperature=0.2)
        response = res.get("content", "")
        usage = res.get("usage", {}) or {}

        # TOKEN/COST tracking
        state.total_tokens += usage.get("total_tokens", 0)
        state.total_cost += calculate_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        # COST GUARDRAIL
        if state.total_cost > state.cost_limit:
            state.abort = True
            state.errors.append(
                ErrorLog(
                    node="reporter_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"Cost exceeded limit: ₹{state.total_cost}",
                )
            )
            state.report = ReportModel(
                title="Research Report",
                sections=[response] if response else ["Cost limit exceeded."],
                citations=citations_list,
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm",
                    "partial": True,
                    "total_tokens": state.total_tokens,
                    "total_cost": state.total_cost,
                    "errors": [e.model_dump() for e in state.errors],
                },
            )
            state.node_execution_count += 1
            return state

        state.used_citation_ids = used_ids
        state.report = ReportModel(
            title="Research Report",
            sections=[response],
            citations=citations_list,
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm",
                "partial": state.synthesis.partial,
                "total_tokens": state.total_tokens,
                "total_cost": state.total_cost,
                "errors": [e.model_dump() for e in state.errors],
            },
        )

        state.node_logs[NodeNames.REPORTER] = {
            "report_generated": True,
            "citations_used": len(sorted_ids),
            "conflicts": len(state.synthesis.conflicts) if state.synthesis.conflicts else 0,
            "total_tokens": state.total_tokens,
            "total_cost": state.total_cost,
        }

        log_node_execution("reporter", {}, {}, start_time)

        state.node_execution_count += 1
        return state

    except Exception as e:
        if state.errors is None:
            state.errors = []
        state.errors.append(
            ErrorLog(
                node="reporter_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}",
            )
        )

        state.report = ReportModel(
            title="Research Report",
            sections=["Critical failure during report generation."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm",
                "partial": True,
                "errors": [err.model_dump() for err in state.errors],
            },
        )

        state.node_logs[NodeNames.REPORTER] = {
            "report_generated": False,
            "citations_used": 0,
            "conflicts": 0,
            "total_tokens": state.total_tokens,
            "total_cost": state.total_cost,
        }

        log_node_execution("reporter", {}, {}, start_time)
        state.node_execution_count += 1
        return state

