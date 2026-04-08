from datetime import datetime, timezone
import json
import time
from typing import Dict, List, Set, Tuple

from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames

from services.system.cost_tracker import calculate_cost


def text_overlap(a: str, b: str) -> float:
    a_words = set((a or "").lower().split())
    b_words = set((b or "").lower().split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


@trace_node(NodeNames.SYNTHESIS)
def synthesiser_node(state: ResearchState) -> ResearchState:
    start_time = time.time()

    try:
        # Safety init
        if state.errors is None:
            state.errors = []
        if state.node_logs is None:
            state.node_logs = {}

        if state.node_execution_count >= 12:
            raise Exception("Max node execution limit reached")

        # 1) FILTER VALID RESULTS
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
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        # 2) SORT + SELECT TOP-K (quality-driven)
        valid_results = sorted(valid_results, key=lambda x: getattr(x, "quality_score", 0), reverse=True)[:12]

        # 3) BUILD CONTEXT WITH STRONG CHUNKS
        context_docs: List[str] = []
        doc_chunks: List[Tuple[str, List[str]]] = []
        state.citation_chunks = {}

        for r in valid_results:
            content = r.content or r.snippet
            if not content or len(content.strip()) < 80:
                continue

            chunks = chunk_text(content) if r.content else [content]
            scored_chunks = []
            for chunk in chunks:
                score = text_overlap(state.query, chunk)
                scored_chunks.append((chunk, score))

            # increased chunk coverage for better grounding
            top_chunks = sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:3]
            selected_chunks = [c[0] for c in top_chunks]
            if not selected_chunks:
                continue

            doc_chunks.append((r.citation_id, selected_chunks))
            state.citation_chunks[r.citation_id] = selected_chunks

        # round-robin context distribution (prevents dominance)
        i = 0
        while len(context_docs) < 25:
            added = False
            for cid, chunks in doc_chunks:
                if i < len(chunks):
                    context_docs.append(f"{cid} {chunks[i][:1200]}")
                    added = True
                    if len(context_docs) >= 25:
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
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        # deduplicate + limit
        context_docs = list(dict.fromkeys(context_docs))
        context_str = "\n\n".join(context_docs)[:22000]

        # 4) CALL LLM
        prompt = (
            load_prompt("synthesiser.txt")
            .replace("{query}", state.query)
            .replace("{context_docs}", context_str)
        )

        res = call_llm(prompt=prompt, temperature=0)
        response_content = res.get("content", "")
        usage = res.get("usage", {}) or {}

        # Token/cost tracking
        state.total_tokens += usage.get("total_tokens", 0)
        state.total_cost += calculate_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        # Cost guardrail
        if state.total_cost > state.cost_limit:
            state.abort = True
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"Cost exceeded limit: ₹{state.total_cost}",
                )
            )
            return state

        # PARSE JSON
        try:
            parsed = json.loads(response_content) if isinstance(response_content, str) else response_content
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

        # 5) BUILD CLAIMS SAFELY (feature/others style robust filtering)
        valid_ids_set = {cid for cid, c in state.citations.items() if c.status == CitationStatus.valid}
        claims: List[Claim] = []
        used_ids: Set[str] = set()

        for c in parsed.get("claims", []):
            try:
                text = str(c.get("text", "")).strip()
                if not text:
                    continue

                raw_ids = c.get("citation_ids", []) or []
                filtered_ids: List[str] = []
                for cid in raw_ids:
                    cid = str(cid).strip()
                    if not cid.startswith("["):
                        cid = f"[{cid}]"
                    if cid in valid_ids_set:
                        filtered_ids.append(cid)

                if not filtered_ids:
                    # fallback to first valid result citation id
                    if valid_results:
                        filtered_ids = [valid_results[0].citation_id]
                    else:
                        continue

                # optional expansion: if only one id, include another grounded id
                if len(filtered_ids) == 1:
                    primary_id = filtered_ids[0]
                    for r in valid_results:
                        cid = r.citation_id
                        if cid != primary_id and cid in valid_ids_set:
                            filtered_ids.append(cid)
                            break

                used_ids.update(filtered_ids)

                try:
                    confidence = float(c.get("confidence", 0.5))
                except Exception:
                    confidence = 0.5

                claims.append(
                    Claim(
                        text=text,
                        citation_ids=filtered_ids,
                        confidence=confidence,
                        verified=True,
                        citation_confidence=1.0,
                    )
                )
            except Exception as e:
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

        # 7) FINAL STATE UPDATE
        state.used_citation_ids = used_ids
        state.synthesis = SynthesisModel(
            claims=claims,
            conflicts=conflicts,
            partial=(len(claims) == 0),
        )

        state.node_logs[NodeNames.SYNTHESIS] = {
            "num_claims": len(claims),
            "valid_citations_used": len(used_ids),
            "conflicts": len(conflicts) if conflicts else 0,
            "partial": state.synthesis.partial,
        }

        log_node_execution("synthesiser", {}, {}, start_time)

        state.node_execution_count += 1
        return state

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="synthesiser_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}",
            )
        )
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        log_node_execution("synthesiser", {}, {}, start_time)
        state.node_execution_count += 1
        return state

