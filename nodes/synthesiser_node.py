from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from models.enums import CitationStatus

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution

import time
import json
import random


@trace_node("synthesiser_node")
def synthesiser_node(state: ResearchState) -> ResearchState:

    start_time = time.time()

    # FILTER VALID RESULTS FIRST
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
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    # SELECT TOP RESULTS
    valid_results = sorted(
        valid_results,
        key=lambda x: getattr(x, "quality_score", 0),
        reverse=True
    )[:6]

    # BUILD CONTEXT
    context_docs = []
    chunk_count = 0

    for r in valid_results:

        content = r.content or r.snippet

        if not content or len(content.strip()) < 100:
            continue

        chunks = chunk_text(content) if r.content else [content]
        chunks = chunks[:2]

        for chunk in chunks:
            if chunk_count >= 12:
                break

            context_docs.append(f"{r.citation_id} {chunk[:1000]}")
            chunk_count += 1

    if not context_docs:
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    random.shuffle(context_docs)

    context_str = "\n\n".join(context_docs)[:15000]

    # LLM
    prompt = load_prompt("synthesiser.txt")\
        .replace("{query}", state.query)\
        .replace("{context_docs}", context_str)

    response = call_llm(prompt=prompt, temperature=0)

    try:
        response = json.loads(response) if isinstance(response, str) else response
    except:
        response = {}

    # BUILD CLAIMS
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

        if not filtered_ids:
            continue

        try:
            confidence = float(c.get("confidence", 0.5))
        except:
            confidence = 0.5

        used_ids.update(filtered_ids)

        claims.append(
            Claim(
                text=text,
                citation_ids=filtered_ids,
                confidence=confidence,
                verified=True,
                citation_confidence=1.0
            )
        )

    state.used_citation_ids = used_ids

    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=response.get("conflicts", []),
        partial=(len(claims) == 0)
    )

    state.node_logs["synthesiser"] = {
        "num_claims": len(claims),
        "valid_citations_used": len(used_ids),
        "partial": state.synthesis.partial
    }

    log_node_execution("synthesiser", {}, {}, start_time)

    return state