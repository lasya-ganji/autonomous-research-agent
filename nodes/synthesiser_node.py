from models.state import ResearchState
from models.synthesis_models import SynthesisModel, Claim

def synthesiser_node(state: ResearchState) -> ResearchState:
    print("Synthesiser Node")

    claims = [
        Claim(
            text="AI improves healthcare efficiency",
            citation_ids=["1"],
            confidence=0.8,
            verified=True,
            citation_confidence=0.8
        )
    ]

    state.synthesis = SynthesisModel(
        claims=claims,
        conflicts=[],
        partial=False
    )

    return state