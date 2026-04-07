from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus, SeverityEnum, ErrorTypeEnum
from models.error_models import ErrorLog

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution
from services.system.cost_tracker import calculate_cost  # ✅ NEW

import time
import json
from datetime import datetime, timezone


@trace_node("synthesiser_node")
def synthesiser_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    try:
        # Safety init
        if state.errors is None:
            state.errors = []

        # -----------------------------
        # 1. FILTER VALID RESULTS
        # -----------------------------
        valid_results = []

        for results in state.search_results.values():
            for r in results:
                citation = state.citations.get(r.citation_id)

                if not citation:
                    continue

                if citation.status != CitationStatus.valid:
                    continue

                valid_results.append(r)

        if not valid_results:
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.WARNING,
                    error_type=ErrorTypeEnum.low_confidence,
                    message="No valid results for synthesis"
                )
            )
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        # -----------------------------
        # 2. SORT + SELECT
        # -----------------------------
        valid_results = sorted(
            valid_results,
            key=lambda x: getattr(x, "quality_score", 0),
            reverse=True
        )[:10]

        # -----------------------------
        # 3. BUILD CONTEXT
        # -----------------------------
        context_docs = []
        chunk_count = 0
        doc_chunks = []

        for r in valid_results:
            content = r.content or r.snippet

            if not content or len(content.strip()) < 80:
                continue

            chunks = chunk_text(content) if r.content else [content]
            chunks = chunks[:2]

            doc_chunks.append((r.citation_id, chunks))

        i = 0
        while chunk_count < 20:
            added = False

            for cid, chunks in doc_chunks:
                if i < len(chunks):
                    context_docs.append(f"{cid} {chunks[i][:1000]}")
                    chunk_count += 1
                    added = True

                    if chunk_count >= 20:
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
                    message="No usable context built"
                )
            )
            state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
            return state

        context_docs = list(dict.fromkeys(context_docs))
        context_str = "\n\n".join(context_docs)[:20000]

        # -----------------------------
        # 4. CALL LLM (UPDATED)
        # -----------------------------
        prompt = load_prompt("synthesiser.txt")\
            .replace("{query}", state.query)\
            .replace("{context_docs}", context_str)

        res = call_llm(prompt=prompt, temperature=0)

        response = res.get("content", "")
        usage = res.get("usage", {})

        # ✅ TOKEN TRACKING
        state.total_tokens += usage.get("total_tokens", 0)

        # ✅ COST TRACKING
        cost = calculate_cost(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0)
        )
        state.total_cost += cost

        # 🚨 COST GUARDRAIL
        if state.total_cost > state.cost_limit:
            state.abort = True

            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.CRITICAL,
                    error_type=ErrorTypeEnum.timeout,
                    message=f"Cost exceeded limit: ₹{state.total_cost}"
                )
            )

            return state

        # -----------------------------
        # PARSE JSON
        # -----------------------------
        try:
            response = json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            state.errors.append(
                ErrorLog(
                    node="synthesiser_node",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    severity=SeverityEnum.ERROR,
                    error_type=ErrorTypeEnum.parsing_error,
                    message=f"LLM JSON parse failed: {str(e)}"
                )
            )
            response = {}

        # -----------------------------
        # 5. BUILD CLAIMS
        # -----------------------------
        claims = []
        used_ids = set()

        valid_ids_set = {
            cid for cid, c in state.citations.items()
            if c.status == CitationStatus.valid
        }

        for c in response.get("claims", []):
            try:
                text = c.get("text", "").strip()
                if not text:
                    continue

                raw_ids = c.get("citation_ids", [])
                filtered_ids = []

                for cid in raw_ids:
                    cid = str(cid).strip()

                    if not cid.startswith("["):
                        cid = f"[{cid}]"

                    if cid in valid_ids_set:
                        filtered_ids.append(cid)

                if not filtered_ids:
                    if valid_results:
                        filtered_ids = [valid_results[0].citation_id]
                    else:
                        continue

                if len(filtered_ids) == 1:
                    primary_id = filtered_ids[0]

                    for r in valid_results:
                        cid = r.citation_id
                        if cid != primary_id and cid not in filtered_ids:
                            filtered_ids.append(cid)
                            break

                used_ids.update(filtered_ids)

                try:
                    confidence = float(c.get("confidence", 0.5))
                except:
                    confidence = 0.5

                claims.append(
                    Claim(
                        text=text,
                        citation_ids=filtered_ids,
                        confidence=confidence,
                        verified=True,
                        citation_confidence=1.0
                    )
                )

            except Exception as e:
                state.errors.append(
                    ErrorLog(
                        node="synthesiser_node",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        severity=SeverityEnum.WARNING,
                        error_type=ErrorTypeEnum.parsing_error,
                        message=f"Claim parsing failed: {str(e)}"
                    )
                )

        # -----------------------------
        # FINAL STATE
        # -----------------------------
        state.used_citation_ids = used_ids

        state.synthesis = SynthesisModel(
            claims=claims,
            conflicts=response.get("conflicts", []),
            partial=(len(claims) == 0)
        )

        print(f"[SYNTHESIS DEBUG] valid={len(valid_results)} used={len(used_ids)} claims={len(claims)}")

        state.node_logs["synthesiser"] = {
            "num_claims": len(claims),
            "valid_citations_used": len(used_ids),
            "partial": state.synthesis.partial
        }

    except Exception as e:
        state.errors.append(
            ErrorLog(
                node="synthesiser_node",
                timestamp=datetime.now(timezone.utc).isoformat(),
                severity=SeverityEnum.CRITICAL,
                error_type=ErrorTypeEnum.parsing_error,
                message=f"Unexpected error: {str(e)}"
            )
        )

    log_node_execution("synthesiser", {}, {}, start_time)

    state.node_execution_count += 1

    return state