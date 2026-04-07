from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution
from config.constants.node_names import NodeNames

import json


def text_overlap(a: str, b: str) -> float:
    a_words = set((a or "").lower().split())
    b_words = set((b or "").lower().split())

    if not a_words or not b_words:
        return 0.0

    return len(a_words & b_words) / len(a_words | b_words)


@trace_node(NodeNames.SYNTHESIS)
def synthesiser_node(state: ResearchState) -> ResearchState:

    # 1. FILTER VALID RESULTS
    valid_results = []

    for results in state.search_results.values():
        for r in results:
            citation = state.citations.get(r.citation_id)
            if citation and citation.status == CitationStatus.valid:
                valid_results.append(r)

    if not valid_results:
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    # 2. SORT + SELECT TOP-K (quality-driven)
    valid_results = sorted(
        valid_results,
        key=lambda x: getattr(x, "quality_score", 0),
        reverse=True
    )[:12]  

    # 3. BUILD CONTEXT WITH STRONG CHUNKS
    context_docs = []
    doc_chunks = []

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
        top_chunks = sorted(
            scored_chunks,
            key=lambda x: x[1],
            reverse=True
        )[:3]

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
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    # deduplicate + limit
    context_docs = list(dict.fromkeys(context_docs))
    context_str = "\n\n".join(context_docs)[:22000]

    # 4. CALL LLM
    prompt = load_prompt("synthesiser.txt") \
        .replace("{query}", state.query) \
        .replace("{context_docs}", context_str)

    try:
        response = call_llm(prompt=prompt, temperature=0)
        response = json.loads(response) if isinstance(response, str) else response
    except Exception:
        # Fallback to partial synthesis
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    # 5. BUILD CLAIMS SAFELY
    claims = []
    used_ids = set()

    valid_ids_set = {
        cid for cid, c in state.citations.items()
        if c.status == CitationStatus.valid
    }

    for c in response.get("claims", []):
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

        # strict grounding enforcement
        if not filtered_ids:
            continue

        try:
            confidence = float(c.get("confidence", 0.5))
        except Exception:
            confidence = 0.5

        claims.append(
            Claim(
                text=text,
                citation_ids=filtered_ids,
                confidence=confidence,
                verified=False,
                citation_confidence=0.0
            )
        )

        used_ids.update(filtered_ids)

    # 6. HANDLE CONFLICTS 
    conflicts = response.get("conflicts", [])

    # 7. FINAL STATE UPDATE
    state.used_citation_ids = used_ids

    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=conflicts,
        partial=(len(claims) == 0)
    )

    print(
        f"[SYNTHESIS DEBUG] valid={len(valid_results)} "
        f"used={len(used_ids)} claims={len(claims)}"
    )

    state.node_logs[NodeNames.SYNTHESIS] = {
        "num_claims": len(claims),
        "valid_citations_used": len(used_ids),
        "conflicts": len(conflicts),
        "partial": state.synthesis.partial
    }

    log_node_execution("synthesiser", {}, {})

    return state