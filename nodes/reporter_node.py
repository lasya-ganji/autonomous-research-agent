from datetime import datetime, timezone
from typing import Dict, Set

from observability.tracing import trace_node

from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from utils.logger import log_node_execution
from config.constants.node_constants.node_names import NodeNames

from services.system.cost_tracker import calculate_cost
from utils.evidence_text import clean_evidence_text, extract_best_excerpt

from config.constants.node_constants.reporter_constants import MIN_REQUIRED_CITATIONS
from config.constants.llm_constants import REPORTER_TEMPERATURE


@trace_node(NodeNames.REPORTER)
def reporter_node(state: ResearchState) -> ResearchState:

    try:
        # SAFETY INIT
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

        # EMPTY SYNTHESIS
        if not state.synthesis or not state.synthesis.claims:
            state.report = ReportModel(
                title="Research Report",
                sections=["No reliable data found."],
                citations=[],
                claim_evidence=[],
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "model": "llm",
                    "partial": True,
                    "errors": [e.model_dump() for e in state.errors],
                },
            )

            state.node_logs[NodeNames.REPORTER] = {
                "report_generated": False,
                "citations_used": 0,
                "claims": 0,
                "errors_count": len(state.errors)
            }

            log_node_execution("reporter_node", {}, {})
            return state

        # VALID CITATIONS
        valid_citations = {
            cid: c for cid, c in state.citations.items()
            if c.status == CitationStatus.valid
        }

        # COLLECT USED IDS
        used_ids: Set[str] = set()
        for claim in state.synthesis.claims:
            for cid in getattr(claim, "citation_ids", []):
                if cid in valid_citations:
                    used_ids.add(cid)

        sorted_ids = sorted(
            used_ids,
            key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0
        )

        id_mapping = {old: f"[{i+1}]" for i, old in enumerate(sorted_ids)}
        state.citation_mapping = id_mapping

        # BUILD SYNTHESIS TEXT
        synthesis_lines = []
        unverified_count = 0

        for claim in state.synthesis.claims:
            valid_ids = [
                cid for cid in getattr(claim, "citation_ids", [])
                if cid in valid_citations
            ]

            if not valid_ids:
                synthesis_lines.append(f"{claim.text} [UNVERIFIED]")
                unverified_count += 1
                continue

            mapped_ids = [id_mapping.get(cid, cid) for cid in valid_ids]
            synthesis_lines.append(f"{claim.text} {' '.join(mapped_ids)}")

        synthesis_text = "\n".join(synthesis_lines)

        # CITATIONS LIST
        citations_list = []
        citations_text_lines = []
        claim_evidence = []
        citation_snippet_map: Dict[str, str] = {}

        for cid in sorted_ids:
            c = valid_citations[cid]
            chunks = state.citation_chunks.get(cid, [])
            snippet = chunks[0] if chunks else ""
            snippet = (snippet[:320] + "...") if len(snippet) > 320 else snippet

            entry = {
                "id": id_mapping[cid],
                "url": c.url,
                "title": getattr(c, "title", ""),
                "quality_score": getattr(c, "quality_score", None),
                "evidence_snippet": snippet,
                "accessed_at": datetime.now().isoformat(),
            }

            citations_list.append(entry)
            citations_text_lines.append(
                f"{entry['id']} {entry['title']} - {entry['url']}"
            )

        citations_text = "\n".join(citations_text_lines)

        for claim in state.synthesis.claims:
            mapped_ids = [id_mapping.get(cid, cid) for cid in getattr(claim, "citation_ids", [])]
            claim_evidence_entries = []
            for ev in getattr(claim, "evidence", []):
                mapped_cid = id_mapping.get(ev.citation_id, ev.citation_id)
                cleaned_snippet = clean_evidence_text(ev.evidence_snippet or "")
                claim_evidence_entries.append(
                    {
                        "citation_id": mapped_cid,
                        "source_title": ev.source_title,
                        "source_url": ev.source_url,
                        "evidence_snippet": cleaned_snippet,
                        "support_score": ev.support_score,
                    }
                )
                if cleaned_snippet and mapped_cid not in citation_snippet_map:
                    citation_snippet_map[mapped_cid] = cleaned_snippet

            claim_evidence.append(
                {
                    "claim_text": claim.text,
                    "support_status": getattr(claim, "support_status", "partially_verified"),
                    "support_reason": getattr(claim, "support_reason", "citation_support_pending"),
                    "confidence": round(getattr(claim, "confidence", 0.0), 3),
                    "citation_confidence": round(getattr(claim, "citation_confidence", 0.0), 3),
                    "citation_ids": mapped_ids,
                    "evidence": claim_evidence_entries,
                }
            )

        # Finalize citation snippets using claim-aligned evidence first, then clean fallback.
        for entry in citations_list:
            cid = entry["id"]
            if cid in citation_snippet_map:
                entry["evidence_snippet"] = citation_snippet_map[cid]
                continue

            original_cid = next((old for old, new in id_mapping.items() if new == cid), None)
            fallback_chunks = state.citation_chunks.get(original_cid, []) if original_cid else []
            raw_fallback = fallback_chunks[0] if fallback_chunks else ""
            fallback_excerpt = extract_best_excerpt("", raw_fallback, max_chars=320)
            entry["evidence_snippet"] = fallback_excerpt or "No readable evidence excerpt available."

        # LLM FORMATTER
        prompt = (
            load_prompt("reporter.txt")
            .replace("{query}", state.query)
            .replace("{synthesis}", synthesis_text)
            .replace("{citations}", citations_text)
        )

        res = call_llm(prompt=prompt, temperature=REPORTER_TEMPERATURE)


        if isinstance(res, dict) and res.get("error"):
            error_type_raw = res.get("error_type", "unknown_error")

            if error_type_raw == "api_error":
                error_type = ErrorTypeEnum.api_error
                severity = SeverityEnum.CRITICAL
                state.api_failure = True
            elif error_type_raw == "timeout_error":
                error_type = ErrorTypeEnum.timeout_error
                severity = SeverityEnum.ERROR
            elif error_type_raw == "network_error":
                error_type = ErrorTypeEnum.network_error
                severity = SeverityEnum.ERROR
            else:
                error_type = ErrorTypeEnum.system_error
                severity = SeverityEnum.ERROR

            state.errors.append(
                ErrorLog(
                    node="reporter_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=severity,
                    error_type=error_type,
                    message=f"LLM failure: {res.get('error')}",
                )
            )

            state.is_partial = True

            state.report = ReportModel(
                title="Research Report",
                sections=["Report generation failed due to LLM error."],
                citations=citations_list,
                claim_evidence=claim_evidence,
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "partial": True,
                    "evidence_metrics": dict(getattr(state, "evidence_metrics", {})),
                    "errors": [e.model_dump() for e in state.errors],
                },
            )

            return state

        response = res.get("content", "")
        usage = res.get("usage", {}) or {}

        print(f"[REPORT] Generated length: {len(response)}")
        print(f"[REPORT] Citations used: {len(used_ids)} | Unverified: {unverified_count}")

        # NODE COST
        node_tokens = usage.get("total_tokens", 0)
        node_cost = calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )

        # GLOBAL COST
        state.total_tokens += node_tokens
        state.total_cost += node_cost

        if state.total_cost > state.cost_limit:
            state.abort = True

            state.errors.append(
                ErrorLog(
                    node="reporter_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.budget_exceeded,
                    message=f"Cost exceeded limit: ₹{state.total_cost}",
                )
            )

            state.report = ReportModel(
                title="Research Report",
                sections=[response] if response else ["Cost limit exceeded."],
                citations=citations_list,
                claim_evidence=claim_evidence,
                metadata={
                    "query": state.query,
                    "timestamp": datetime.now().isoformat(),
                    "partial": True,
                    "total_tokens": state.total_tokens,
                    "total_cost": state.total_cost,
                    "evidence_metrics": dict(getattr(state, "evidence_metrics", {})),
                    "errors": [e.model_dump() for e in state.errors],
                },
            )

            return state

        partial_flag = (
            state.is_partial or
            state.synthesis.partial or
            state.abort or
            len(used_ids) < MIN_REQUIRED_CITATIONS
        )

        if partial_flag:
            state.is_partial = True

        # FINAL REPORT
        state.report = ReportModel(
            title="Research Report",
            sections=[response] if response else ["No structured report generated."],
            citations=citations_list,
            claim_evidence=claim_evidence,
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "partial": partial_flag,
                "total_tokens": state.total_tokens,
                "total_cost": state.total_cost,
                "evidence_metrics": dict(getattr(state, "evidence_metrics", {})),
                "errors": [e.model_dump() for e in state.errors],
            },
        )

        # OBSERVABILITY
        node_name = NodeNames.REPORTER
        existing_log = state.node_logs.get(node_name, {})

        existing_log.update({
            "report_generated": True,
            "citations_used": len(used_ids),
            "claims": len(state.synthesis.claims),
            "unverified_claims": unverified_count,
            "partial": partial_flag,
            "errors_count": len(state.errors),
            "node_tokens": node_tokens,
            "node_cost": round(node_cost, 4),
            "total_cost": round(state.total_cost, 4)
        })

        state.node_logs[node_name] = existing_log

        log_node_execution(
            "reporter_node",
            {"claims": len(state.synthesis.claims)},
            {"citations": len(used_ids)}
        )

        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="reporter_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=str(e),
            )
        )

        state.report = ReportModel(
            title="Research Report",
            sections=["Critical failure during report generation."],
            citations=[],
            claim_evidence=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "partial": True,
                "errors": [err.model_dump() for err in state.errors],
            },
        )

        state.node_logs[NodeNames.REPORTER] = {
            "report_generated": False
        }

        log_node_execution("reporter_node", {}, {})

        return state