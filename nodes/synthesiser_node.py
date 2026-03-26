from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
import json


def synthesiser_node(state: ResearchState) -> ResearchState:
    print("Synthesiser Node")

    # Build structured context_docs
    context_docs = []

    for step_id, results in state.search_results.items():
        for r in results[:2]:  # top 2 per step
            if r.snippet:
                context_docs.append({
                    "citation_id": r.citation_id,
                    "content": r.snippet
                })

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

    return state