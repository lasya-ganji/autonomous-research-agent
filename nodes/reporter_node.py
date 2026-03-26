from models.state import ResearchState
from models.report_models import ReportModel
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from datetime import datetime

@trace_node("reporter_node")
def reporter_node(state: ResearchState) -> ResearchState:
    print("Reporter Node")

    # Prepare synthesis text
    synthesis_text = "\n".join(
        [f"- {claim.text}" for claim in state.synthesis.claims]
    )

    # Prepare citations
    citations_list = list(state.citations.values())[:5]

    citations_text = "\n".join(
        [f"[{i+1}] {c.url}" for i, c in enumerate(citations_list)]
    )

    # Load prompt
    prompt_template = load_prompt("reporter.txt")

    prompt = prompt_template.format(
        query=state.query,
        synthesis=synthesis_text,
        citations=citations_text
    )

    # Call LLM
    response = call_llm(prompt=prompt, temperature=0.3)

    # Store clean report
    state.report = ReportModel(
        title="Research Report",
        sections=[response],   
        citations=list(state.citations.keys()),
        metadata={
            "query": state.query,
            "timestamp": datetime.now().isoformat(),
            "model": "llm"
        }
    )

    return state