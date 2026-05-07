from datetime import datetime, timezone
import json
from typing import List

from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution
from config.constants.node_constants.node_names import NodeNames

from services.system.cost_tracker import calculate_cost

from config.constants.node_constants.synthesis_constants import (
    MAX_SYNTHESIS_RESULTS,
    MAX_CHUNKS_PER_DOC,
    MAX_CONTEXT_DOCS,
    MAX_CONTEXT_LENGTH,
    MAX_CHUNK_LENGTH,
    MIN_CONTENT_LENGTH,
    MIN_CITATIONS_REQUIRED
)
from config.constants.llm_constants import SYNTHESISER_TEMPERATURE


def text_overlap(a: str, b: str) -> float:
    a_words = set((a or "").lower().split())
    b_words = set((b or "").lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


@trace_node(NodeNames.SYNTHESIS)
def synthesiser_node(state: ResearchState) -> ResearchState:

    try:
        # SAFETY INIT
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

        # FILTER VALID RESULTS
        valid_results = []
        for results in state.search_results.values():
            for r in results:
                citation = state.citations.get(r.citation_id)
                if citation and citation.status == CitationStatus.valid:
                    valid_results.append(r)

        if not valid_results:
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.low_confidence,
                    message="No valid search results for synthesis",
                )
            )

            state.is_partial = True
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        # SELECT TOP RESULTS
        valid_results = sorted(
            valid_results,
            key=lambda x: getattr(x, "quality_score", 0),
            reverse=True
        )[:MAX_SYNTHESIS_RESULTS]

        # BUILD CONTEXT
        context_docs = []
        doc_chunks = []
        state.citation_chunks = {}

        for r in valid_results:
            content = r.content or r.snippet
            if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
                continue

            chunks = chunk_text(content) if r.content else [content]

            scored_chunks = [(c, text_overlap(state.query, c)) for c in chunks]
            top_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:MAX_CHUNKS_PER_DOC]

            selected_chunks = [c[0] for c in top_chunks if c[0]]
            if not selected_chunks:
                continue

            doc_chunks.append((r.citation_id, selected_chunks))
            state.citation_chunks[r.citation_id] = selected_chunks

        i = 0
        while len(context_docs) < MAX_CONTEXT_DOCS:
            added = False
            for cid, chunks in doc_chunks:
                if i < len(chunks):
                    context_docs.append(f"{cid} {chunks[i][:MAX_CHUNK_LENGTH]}")
                    added = True
                    if len(context_docs) >= MAX_CONTEXT_DOCS:
                        break
            if not added:
                break
            i += 1

        if not context_docs:
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.low_confidence,
                    message="No usable context built for synthesis",
                )
            )

            state.is_partial = True
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        context_docs = list(dict.fromkeys(context_docs))
        context_str = "\n\n".join(context_docs)[:MAX_CONTEXT_LENGTH]

        print(f"[SYNTHESIS] Context docs built: {len(context_docs)}")

        # LLM CALL
        prompt = (
            load_prompt("synthesiser.txt")
            .replace("{query}", state.query)
            .replace("{context_docs}", context_str)
        )

        res = call_llm(prompt=prompt, temperature=SYNTHESISER_TEMPERATURE)

        # HANDLE LLM ERRORS
        if isinstance(res, dict) and res.get("error"):
            raw_type = res.get("error_type", "unknown_error")

            if raw_type == "api_error":
                error_type = ErrorTypeEnum.api_error
                severity = SeverityEnum.CRITICAL
                state.api_failure = True
            elif raw_type == "timeout_error":
                error_type = ErrorTypeEnum.timeout_error
                severity = SeverityEnum.ERROR
            elif raw_type == "network_error":
                error_type = ErrorTypeEnum.network_error
                severity = SeverityEnum.ERROR
            else:
                error_type = ErrorTypeEnum.system_error
                severity = SeverityEnum.ERROR

            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=severity,
                    error_type=error_type,
                    message=f"LLM failure: {res.get('error')}",
                )
            )

            state.is_partial = True
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        response_content = res.get("content", "") or ""
        usage = res.get("usage", {}) or {}

        # EMPTY RESPONSE GUARD
        if not response_content.strip():
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.parsing_error,
                    message="Empty response from LLM",
                )
            )

            state.is_partial = True
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        # COST TRACKING
        node_tokens = usage.get("total_tokens", 0)
        node_cost = calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )

        state.total_tokens += node_tokens
        state.total_cost += node_cost

        if state.total_cost > state.cost_limit:
            state.abort = True
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.budget_exceeded,
                    message=f"Cost exceeded: ₹{state.total_cost}",
                )
            )
            return state

        # PARSE RESPONSE (ROBUST)
        try:
            parsed = json.loads(response_content)
        except Exception:
            try:
                start = response_content.find("{")
                end = response_content.rfind("}") + 1
                parsed = json.loads(response_content[start:end])
            except Exception as e:
                state.errors.append(
                    ErrorLog(
                        node="synthesiser_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.ERROR,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"LLM JSON parse failed: {str(e)}",
                    )
                )
                parsed = {}

        # CLAIM PROCESSING
        valid_ids_set = {
            cid for cid, c in state.citations.items()
            if c.status == CitationStatus.valid
        }

        claims = []
        used_ids = set()
        dropped_claims = 0

        for c in parsed.get("claims", []):
            try:
                text = str(c.get("text", "")).strip()
                if not text:
                    continue

                raw_ids = c.get("citation_ids", []) or []
                filtered_ids = []

                for cid in raw_ids:
                    cid = str(cid).strip()
                    if not cid.startswith("["):
                        cid = f"[{cid}]"

                    if cid in valid_ids_set:
                        filtered_ids.append(cid)

                has_supporting_citations = len(filtered_ids) > 0
                if has_supporting_citations:
                    used_ids.update(filtered_ids)

                confidence = float(c.get("confidence", 0.5))
                confidence = max(0.0, min(confidence, 1.0))

                claims.append(
                    Claim(
                        text=text,
                        citation_ids=filtered_ids,
                        confidence=confidence,
                        verified=has_supporting_citations,
                        citation_confidence=1.0 if has_supporting_citations else 0.0,
                        support_status="partially_verified" if not has_supporting_citations else "verified",
                        support_reason="missing_initial_citation_support" if not has_supporting_citations else "initial_citation_match",
                    )
                )

            except Exception as e:
                dropped_claims += 1
                state.errors.append(
                    ErrorLog(
                        node="synthesiser_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"Claim parsing failed: {str(e)}",
                    )
                )

        conflicts = parsed.get("conflicts", [])

        evaluator_failed = state.node_logs.get(NodeNames.EVALUATOR, {}).get("all_failed", False)

        partial_flag = (
            state.is_partial or
            len(claims) == 0 or
            len(used_ids) < MIN_CITATIONS_REQUIRED or
            evaluator_failed
        )

        if partial_flag:
            state.is_partial = True

        state.used_citation_ids = used_ids
        state.synthesis = SynthesisModel(
            claims=claims,
            conflicts=conflicts,
            partial=partial_flag,
        )

        # LOGGING
        existing_log = state.node_logs.get(NodeNames.SYNTHESIS, {})
        existing_log.update({
            "num_claims": len(claims),
            "valid_citations_used": len(used_ids),
            "conflicts": len(conflicts),
            "partial": partial_flag,
            "context_docs": len(context_docs),
            "dropped_claims": dropped_claims,
            "errors_count": len(state.errors),
            "node_tokens": node_tokens,
            "node_cost": round(node_cost, 4),
            "total_cost": round(state.total_cost, 4)
        })
        state.node_logs[NodeNames.SYNTHESIS] = existing_log

        log_node_execution(
            "synthesiser_node",
            {"context_docs": len(context_docs)},
            {"claims": len(claims)}
        )

        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="synthesiser_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.system_error,
                message=f"Unexpected error: {str(e)}",
            )
        )

        state.is_partial = True
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)

        log_node_execution(
            "synthesiser_node",
            {},
            {"error": str(e)}
        )

        return state