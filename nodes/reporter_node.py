from datetime import datetime
from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node


@trace_node("reporter_node")
def reporter_node(state: ResearchState) -> ResearchState:

    if not state.synthesis or not state.synthesis.claims:
        state.report = ReportModel(
            title="Research Report",
            sections=["No reliable data found."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm"
            }
        )
        return state

    # valid citations
    valid_citations = {
        cid: c for cid, c in state.citations.items()
        if c.status == CitationStatus.valid
    }

    # build synthesis text
    synthesis_lines = []
    used_ids = set()

    for claim in state.synthesis.claims:

        valid_ids = [
            cid for cid in claim.citation_ids
            if cid in valid_citations
        ]

        if not valid_ids:
            continue

        used_ids.update(valid_ids)

        synthesis_lines.append(
            f"{claim.text} {' '.join(valid_ids)}"
        )

    if not synthesis_lines:
        state.report = ReportModel(
            title="Research Report",
            sections=["No valid citations available."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm"
            }
        )
        return state

    synthesis_text = "\n".join(synthesis_lines)

    citations_text = "\n".join(
        [f"{cid} {valid_citations[cid].url}" for cid in used_ids]
    )

    prompt = load_prompt("reporter.txt")\
        .replace("{query}", state.query)\
        .replace("{synthesis}", synthesis_text)\
        .replace("{citations}", citations_text)

    response = call_llm(prompt=prompt, temperature=0.2)

    state.used_citation_ids = used_ids

    state.report = ReportModel(
        title="Research Report",
        sections=[response],
        citations=list(used_ids),
        metadata={
            "query": state.query,
            "timestamp": datetime.now().isoformat(),
            "model": "llm"
        }
    )

    state.node_logs["reporter"] = {
        "report_generated": True,
        "citations_used": len(used_ids)
    }

    return state