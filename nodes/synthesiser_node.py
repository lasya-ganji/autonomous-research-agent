from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution
import time

import json


@trace_node("synthesiser_node")
def synthesiser_node(state: ResearchState) -> ResearchState:
    print("Synthesiser Node")

    start_time = time.time()

    input_data = {
        "query": state.query[:100],  # avoid large logs
        "num_steps": len(state.search_results),
    }
    
    # CONFIG (tuned for safety + quality)
    TOP_K_RESULTS = 6              # best sources only
    MAX_CHUNKS = 12               # total chunks
    MAX_CHUNK_SIZE = 1000         # per chunk
    MAX_CONTEXT_CHARS = 15000     # final prompt safety cap

    # Step 1: Flatten + sort results by quality
    all_results = []
    for results in state.search_results.values():
        all_results.extend(results)

    if not all_results:
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    all_results = sorted(
        all_results,
        key=lambda x: getattr(x, "quality_score", 0),
        reverse=True
    )[:TOP_K_RESULTS]

    # Step 2: Build context_docs with controlled chunking
    context_docs = []
    chunk_count = 0

    for r in all_results:
        content = r.content or r.snippet or ""
        if not content:
            continue

        # chunk if large content
        chunks = chunk_text(content) if r.content else [content]

        for chunk in chunks:
            if chunk_count >= MAX_CHUNKS:
                break

            chunk = chunk[:MAX_CHUNK_SIZE]

            context_docs.append({
                "citation_id": r.citation_id,
                "content": chunk
            })

            chunk_count += 1

        if chunk_count >= MAX_CHUNKS:
            break

    # Step 3: Handle empty context
    if not context_docs:
        state.synthesis = SynthesisModel(
            claims=[],
            conflicts=[],
            partial=True
        )
        return state

    # Step 4: Build safe prompt (NO indent)
    context_str = json.dumps(context_docs)

    # hard safety cap
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS]

    prompt_template = load_prompt("synthesiser.txt")

    prompt = prompt_template.format(
        query=state.query,
        context_docs=context_str
    )

    # Step 5: Call LLM
    response = call_llm(prompt=prompt, temperature=0)

    # Step 6: Safe JSON parsing
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception as e:
            print("[SYNTHESIS ERROR] JSON parse failed:", e)
            response = {}

    print("[SYNTHESIS RAW]:", response)

    # Step 7: Extract claims
    claims = []

    if isinstance(response, dict) and "claims" in response:
        for c in response["claims"]:
            claims.append(
                Claim(
                    text=c.get("text", ""),
                    citation_ids=c.get("citation_ids", []),
                    confidence=c.get("confidence", 0.5),
                    verified=True,
                    citation_confidence=c.get("confidence", 0.5)
                )
            )

    # Step 8: Store synthesis
    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=response.get("conflicts", []),
        partial=response.get("partial", False) or len(claims) == 0
    )

    # Step 9: Logging (structured)
    state.node_logs["synthesiser"] = {
        "num_claims": len(state.synthesis.claims),
        "used_sources": len(all_results),
        "chunks_used": len(context_docs),
        "partial": state.synthesis.partial,
        "claims": [
            {
                "text": c.text,
                "confidence": c.confidence,
                "citations": c.citation_ids
            }
            for c in state.synthesis.claims
        ]
    }
    
    output_data = {
        "num_claims": len(state.synthesis.claims),
        "partial": state.synthesis.partial,
        "used_sources": len(all_results),
        "chunks_used": len(context_docs)
    }

    log_node_execution("synthesiser", input_data, output_data, start_time)

    return state