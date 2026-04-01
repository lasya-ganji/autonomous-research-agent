import json
import traceback
import numpy as np

from datetime import datetime
from models.state import ResearchState
from models.report_models import ReportModel
from models.enums import CitationStatus
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from observability.tracing import trace_node
from services.retrieval.embedding_service import get_embedding
from app.dependencies import semantic_cache


@trace_node("reporter_node")
def reporter_node(state: ResearchState) -> ResearchState:
    print("Reporter Node")
        # 🔥 ADD THIS
    if state.cache_hit:
        print("⚡ Returning cached final response")
        return state

    # Step 1: Keep only VALID citations
    valid_citations = [
        c for c in state.citations.values()
        if c.status == CitationStatus.valid
    ]

    # Step 2: Keep only citations actually used in synthesis
    citations_list = []
    if state.synthesis and state.synthesis.claims:
        used_ids = set()
        for claim in state.synthesis.claims:
            for cid in claim.citation_ids:
                if cid in [c.citation_id for c in valid_citations] and cid not in used_ids:
                    citations_list.append(next(c for c in valid_citations if c.citation_id == cid))
                    used_ids.add(cid)

    # Step 3: Map citation_id -> index for inline references
    citation_index_map = {c.citation_id: idx + 1 for idx, c in enumerate(citations_list)}

    # Step 4: Build structured synthesis for LLM (only with valid citations)
    synthesis_structured = []
    if state.synthesis and state.synthesis.claims:
        for claim in state.synthesis.claims:
            indices = [str(citation_index_map[cid]) for cid in claim.citation_ids if cid in citation_index_map]
            if not indices:
                continue  # skip claims with no valid citations
            synthesis_structured.append({
                "text": claim.text,
                "citations": indices
            })

    # Step 5: Convert structured synthesis to text for LLM
    synthesis_text = json.dumps(synthesis_structured, indent=2)

    # Step 6: Prepare citations section (only valid & used)
    citations_text = "\n".join([f"[{i+1}] {c.url}" for i, c in enumerate(citations_list)])

    # Step 7: Load reporter prompt
    prompt_template = load_prompt("reporter.txt")
    prompt = prompt_template.format(
        query=state.query,
        synthesis=synthesis_text,
        citations=citations_text
    )

    # Step 8: Call LLM
    response = call_llm(prompt=prompt, temperature=0.3)

    # Step 9: Store final report
    state.report = ReportModel(
        title="Research Report",
        sections=[response],
        citations=[c.citation_id for c in citations_list],  # only valid ones
        metadata={
            "query": state.query,
            "timestamp": datetime.now().isoformat(),
            "model": "llm"
        }
    )

    # Step 10: Logging
    state.node_logs["reporter"] = {
        "report_generated": True,
        "num_sections": len(state.report.sections),
        "citations_used": len(citations_list)
    }
    print("[CACHE DEBUG] entered cache block")

    try:
        if not hasattr(state, "query") or not state.query:
            print("❌ missing query")
            return state

        if not semantic_cache:
            print("❌ cache not initialized")
            return state

        result_text = json.dumps(state.report.model_dump(), default=str)

        if not result_text:
            print("❌ no result_text found")
            return state

        embedding = get_embedding(state.query)

        if not embedding or len(embedding) != semantic_cache.dim:
            print("❌ invalid embedding:", embedding)
            return state

        emb = np.array([embedding], dtype="float32")

        if len(emb.shape) != 2:
            print("❌ invalid embedding shape:", emb.shape)
            return state

        print("[CACHE DEBUG] about to insert into FAISS")

        semantic_cache.add(
            query=state.query,
            embedding=emb,
            result=result_text,
            quality_score=getattr(state, "overall_confidence", 0.7),
            citation_confidence=getattr(state, "overall_confidence", 0.7),
            is_failed=False,
        )

        print("✅ CACHE STORED SUCCESSFULLY")

    except Exception as e:
        print("🔥 CACHE STORE ERROR OCCURRED")
        print("TYPE:", type(e))
        print("REPR:", repr(e))
        print("ARGS:", e.args)
        traceback.print_exc()

    return state