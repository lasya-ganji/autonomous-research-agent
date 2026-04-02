from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
from utils.logger import log_node_execution

import time
import json
import random

@trace_node("synthesiser_node")
def synthesiser_node(state: ResearchState, llm=None) -> ResearchState:
    print("Synthesiser Node")

    # Skip if flagged
    if getattr(state, "skip_remaining", False):
        print("⚡ Skipping synthesiser (cache hit)")
        return state

    start_time = time.time()

    input_data = {
        "query": state.query[:100],
        "num_steps": len(state.search_results),
    }

    # Config
    MAX_CHUNKS = 12
    MAX_CHUNK_SIZE = 1000
    MAX_CONTEXT_CHARS = 15000
    RESULTS_PER_STEP = 2
    MAX_CHUNKS_PER_SOURCE = 2

    # Step 1: Select top results per step
    selected_results = []
    for step_id, results in state.search_results.items():
        if not results:
            continue

        # sort by quality_score if available
        def score_key(r):
            if hasattr(r, "quality_score"):
                return r.quality_score
            elif isinstance(r, dict):
                return r.get("quality_score", 0)
            else:
                return 0

        top_results = sorted(results, key=score_key, reverse=True)[:RESULTS_PER_STEP]
        selected_results.extend(top_results)

    if not selected_results:
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    # Step 2: Build balanced context
    context_docs = []
    chunk_count = 0

    for r in selected_results:
        # Determine content
        content = None
        title = "No Title"
        citation_id = "N/A"

        if hasattr(r, "content"):
            content = r.content or r.snippet
            title = getattr(r, "title", "No Title")
            citation_id = getattr(r, "citation_id", "N/A")
        elif isinstance(r, dict):
            content = r.get("content") or r.get("snippet")
            title = r.get("title", "No Title")
            citation_id = r.get("citation_id", "N/A")
        elif isinstance(r, str):
            content = r

        if not content or len(content.strip()) < 25:
            continue

        chunks = chunk_text(content) if hasattr(r, "content") else [content]
        chunks = chunks[:MAX_CHUNKS_PER_SOURCE]

        for chunk in chunks:
            if chunk_count >= MAX_CHUNKS:
                break
            chunk = chunk[:MAX_CHUNK_SIZE]
            context_docs.append(f"[{citation_id}] {title}. {chunk}")
            chunk_count += 1

        if chunk_count >= MAX_CHUNKS:
            break

    if not context_docs:
        state.synthesis = SynthesisModel(claims=[], conflicts=[], partial=True)
        return state

    random.shuffle(context_docs)
    context_str = "\n\n".join(context_docs)
    if len(context_str) > MAX_CONTEXT_CHARS:
        context_str = context_str[:MAX_CONTEXT_CHARS]

    # Step 3: Build prompt
    prompt_template = load_prompt("synthesiser.txt")
    prompt = prompt_template.replace("{query}", state.query)
    prompt = prompt.replace("{context_docs}", context_str)

    # Step 4: Call LLM
    if llm:
        response = llm.generate(prompt)
    else:
        response = call_llm(...)

    # Step 5: Parse response
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception as e:
            print("[SYNTHESIS ERROR] JSON parse failed:", e)
            response = {}

    print("[SYNTHESIS RAW]:", response)
    print("[DEBUG CONTEXT SAMPLE]:", context_docs[:2])

    # Step 6: Extract claims
    claims = []
    if isinstance(response, dict):
        raw_claims = response.get("claims", [])
        for c in raw_claims:
            if not isinstance(c, dict):
                continue

            text = c.get("text", "").strip()
            citation_ids = c.get("citation_ids", [])
            if not text or not citation_ids:
                continue

            try:
                confidence = float(c.get("confidence", 0.5))
            except Exception:
                confidence = 0.5

            claims.append(
                Claim(
                    text=text,
                    citation_ids=citation_ids,
                    confidence=confidence,
                    verified=True,
                    citation_confidence=1.0
                )
            )

    # Step 7: Store synthesis
    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=response.get("conflicts", []) if isinstance(response, dict) else [],
        partial=(len(claims) == 0)
    )

    # Step 8: Node logs
    if state.node_logs is None:
        state.node_logs = {}
    state.node_logs["synthesiser"] = {
        "num_claims": len(claims),
        "used_sources": len(selected_results),
        "chunks_used": len(context_docs),
        "partial": state.synthesis.partial,
        "claims": [
            {"text": c.text, "confidence": c.confidence, "citations": c.citation_ids}
            for c in claims
        ]
    }

    # Step 9: Logger
    output_data = {
        "num_claims": len(claims),
        "partial": state.synthesis.partial,
        "used_sources": len(selected_results),
        "chunks_used": len(context_docs)
    }
    log_node_execution("synthesiser", input_data, output_data, start_time)

    return state