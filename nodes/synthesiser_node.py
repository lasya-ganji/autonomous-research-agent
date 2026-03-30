from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from utils.chunking import chunk_text
import json

@trace_node("synthesiser_node")
def synthesiser_node(state: ResearchState) -> ResearchState:
    print("Synthesiser Node")

    # Build structured context_docs
    context_docs = []
    MAX_CHUNKS = 25
    chunk_count = 0
    step_map = {
    step.step_id: step.priority
    for step in state.research_plan
}

    for step_id, results in state.search_results.items():

        priority = step_map.get(step_id, 3)

    #Chunk allocation based on priority
        if priority == 1:
            max_chunks_per_step = 12
        elif priority == 2:
            max_chunks_per_step = 8
        else:
            max_chunks_per_step = 5

        step_chunk_count = 0

        for r in results:

            content = r.content or r.snippet

            if content:

                #If it's a snippet → don't chunk
                if getattr(r, "snippet", False):
                    chunks = [content]

                #If it's full content → chunk it
                else:
                    chunks = chunk_text(content)

                for chunk in chunks:

                    if chunk_count >= MAX_CHUNKS:
                        break

                    if step_chunk_count >= max_chunks_per_step:
                        break

                    context_docs.append({
                        "citation_id": r.citation_id,
                        "content": chunk
                    })

                    chunk_count += 1
                    step_chunk_count += 1

            if chunk_count >= MAX_CHUNKS:
                break
    # Handle empty
    if not context_docs:
        state.synthesis = SynthesisModel(
            claims=[],
            conflicts=[],
            partial=True
        )
        return state

    # Load prompt
    prompt_template = load_prompt("synthesiser.txt")

    prompt = prompt_template.format(
        query=state.query,
        context_docs=json.dumps(context_docs, indent=2)
    )

    # Call LLM
    response = call_llm(
        prompt=prompt,
        temperature=0
    )
    
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except Exception as e:
            print("[SYNTHESIS ERROR] Failed to parse JSON:", e)
            response = {}

    print("[SYNTHESIS RAW]:", response)

    # Parse response
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

    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=response.get("conflicts", []),
        partial=response.get("partial", False)
    )
    if state.node_logs is None:
        state.node_logs = {}

    state.node_logs["synthesiser"] = {
        "num_claims": len(state.synthesis.claims),
        "claims": [
            {
                "text": c.text,
                "confidence": c.confidence,
                "citations": c.citation_ids
            }
            for c in state.synthesis.claims
        ]
    }

    return state