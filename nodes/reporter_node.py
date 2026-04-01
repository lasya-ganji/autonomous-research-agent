from datetime import datetime
from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node


@trace_node("reporter_node")
def reporter_node(state: ResearchState) -> ResearchState:

    # NO DATA CASE
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

    # VALID CITATIONS
    valid_citations = {
        cid: c for cid, c in state.citations.items()
        if c.status == CitationStatus.valid
    }

    # COLLECT ALL CLAIM IDS
    all_claim_ids = set()

    for claim in state.synthesis.claims:
        for cid in claim.citation_ids:
            if cid in valid_citations:
                all_claim_ids.add(cid)

    # safe numeric sort
    sorted_ids = sorted(
        all_claim_ids,
        key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0
    )

    # CREATE MAPPING
    id_mapping = {old: f"[{i+1}]" for i, old in enumerate(sorted_ids)}

    # store mapping for UI
    state.citation_mapping = id_mapping

    # BUILD SYNTHESIS
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

        # map to sequential IDs
        mapped_ids = [id_mapping.get(cid, cid) for cid in valid_ids]

        synthesis_lines.append(
            f"{claim.text} {' '.join(mapped_ids)}"
        )

    # FALLBACK
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

    # CITATIONS TEXT
    citations_text = "\n".join(
        [
            f"{id_mapping[cid]} {valid_citations[cid].url}"
            for cid in sorted_ids
        ]
    )

    # LLM
    prompt = load_prompt("reporter.txt")\
        .replace("{query}", state.query)\
        .replace("{synthesis}", synthesis_text)\
        .replace("{citations}", citations_text)

    response = call_llm(prompt=prompt, temperature=0.2)

    # STORE STATE
    state.used_citation_ids = set(sorted_ids)

    state.report = ReportModel(
        title="Research Report",
        sections=[response],
        citations=sorted_ids,  
        metadata={
            "query": state.query,
            "timestamp": datetime.now().isoformat(),
            "model": "llm"
        }
    )

    state.node_logs["reporter"] = {
        "report_generated": True,
        "citations_used": len(sorted_ids)
    }

    return state