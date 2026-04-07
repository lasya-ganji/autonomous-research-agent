from datetime import datetime
from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus

from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from config.constants.node_names import NodeNames


@trace_node(NodeNames.REPORTER)
def reporter_node(state: ResearchState) -> ResearchState:

    if not state.synthesis or not state.synthesis.claims:
        state.report = ReportModel(
            title="Research Report",
            sections=["No reliable data found."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm",
                "partial": True,
                "errors": state.errors
            }
        )
        return state

    valid_citations = {
        cid: c for cid, c in state.citations.items()
        if c.status == CitationStatus.valid
    }

    all_claim_ids = set()
    for claim in state.synthesis.claims:
        for cid in claim.citation_ids:
            if cid in valid_citations:
                all_claim_ids.add(cid)

    sorted_ids = sorted(
        all_claim_ids,
        key=lambda x: int(x.strip("[]")) if x.strip("[]").isdigit() else 0
    )

    id_mapping = {old: f"[{i+1}]" for i, old in enumerate(sorted_ids)}
    state.citation_mapping = id_mapping

    synthesis_lines = []
    used_ids = set()

    for claim in state.synthesis.claims:

        valid_ids = [cid for cid in claim.citation_ids if cid in valid_citations]

        if not valid_ids:
            synthesis_lines.append(
                f"{claim.text} [UNVERIFIED]"
            )
            continue

        used_ids.update(valid_ids)

        mapped_ids = [id_mapping.get(cid, cid) for cid in valid_ids]

        synthesis_lines.append(
            f"{claim.text} {' '.join(mapped_ids)} "
            f"(confidence: {round(claim.citation_confidence, 2)}, verified: {claim.verified})"
        )

    if not synthesis_lines:
        state.report = ReportModel(
            title="Research Report",
            sections=["No valid citations available."],
            citations=[],
            metadata={
                "query": state.query,
                "timestamp": datetime.now().isoformat(),
                "model": "llm",
                "partial": True,
                "errors": state.errors
            }
        )
        return state

    synthesis_text = "\n".join(synthesis_lines)

    # FULL CITATION METADATA 
    citations_list = []
    citations_text_lines = []

    for cid in sorted_ids:
        c = valid_citations[cid]

        citation_entry = {
            "id": id_mapping[cid],
            "url": c.url,
            "title": getattr(c, "title", ""),
            "quality_score": getattr(c, "quality_score", None),
            "accessed_at": datetime.now().isoformat()
        }

        citations_list.append(citation_entry)

        citations_text_lines.append(
            f"{citation_entry['id']} {citation_entry['title']} - {citation_entry['url']}"
        )

    citations_text = "\n".join(citations_text_lines)

    # CONFLICTS SECTION
    conflicts_text = ""
    if state.synthesis.conflicts:
        conflicts_text = "\n\nConflicts:\n" + "\n".join(state.synthesis.conflicts)

    prompt = load_prompt("reporter.txt") \
        .replace("{query}", state.query) \
        .replace("{synthesis}", synthesis_text + conflicts_text) \
        .replace("{citations}", citations_text)

    response = call_llm(prompt=prompt, temperature=0.2)

    state.used_citation_ids = used_ids

    state.report = ReportModel(
        title="Research Report",
        sections=[response],
        citations=citations_list,
        metadata={
            "query": state.query,
            "timestamp": datetime.now().isoformat(),
            "model": "llm",
            "partial": state.synthesis.partial,
            "errors": state.errors
        }
    )

    state.node_logs[NodeNames.REPORTER] = {
        "report_generated": True,
        "citations_used": len(sorted_ids),
        "conflicts": len(state.synthesis.conflicts)
    }

    return state