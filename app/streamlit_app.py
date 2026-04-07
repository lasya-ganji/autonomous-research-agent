import os
import logging
import warnings
import time

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")

import streamlit as st
from agent_runner import run_agent
from models.enums import CitationStatus


# ---------------- NODE NAME CONTRACT ----------------

NODE_KEYS = {
    "REPORT": "REPORTER",
    "PLAN": "PLANNER",
    "SEARCH": "SEARCHER",
    "EVALUATION": "EVALUATOR",
    "SYNTHESIS": "SYNTHESIS",
    "CITATION": "CITATION_MANAGER",
}


# ---------------- UTIL ----------------

def safe_get(obj, key, default=None):
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


def get_node_time(node_logs, node_key):
    node = node_logs.get(node_key, {})
    trace = node.get("_trace", {})
    return round(trace.get("duration_s", 0), 2)


def render_report(content):
    st.markdown(content)


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Autonomous Research Agent",
    layout="wide"
)

st.title("Autonomous Research Agent")
st.markdown("Generate structured, citation-backed research reports.")

query = st.text_input("Research Query", placeholder="Enter your query here...")
run_button = st.button("Run Research")


# ---------------- RUN ----------------

if run_button:

    if not query.strip():
        st.warning("Please enter a valid query.")
        st.stop()

    start_time = time.time()

    with st.spinner("Running research agent..."):
        result = run_agent(query)

    total_time = round(time.time() - start_time, 2)

    if hasattr(result, "dict"):
        result = result.dict()

    st.success(f"Research completed in {total_time}s")

    node_logs = result.get("node_logs", {})

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Report", "Plan", "Search", "Evaluation", "Synthesis", "Citations", "Debug"
    ])

    # ---------------- REPORT ----------------
    with tab1:
        report = result.get("report")

        if report:
            sections = getattr(report, "sections", [])
            if sections:
                render_report(sections[0])
            else:
                st.warning("Report content is empty.")
        else:
            st.error("No report generated.")

        st.divider()
        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['REPORT'])}s")


    # ---------------- PLAN ----------------
    with tab2:
        st.subheader("Research Plan")

        plan = result.get("research_plan")

        if plan:
            for step in plan:
                st.markdown(f"**Step {safe_get(step,'step_id')}**")
                st.write(safe_get(step, "question"))
                st.caption(f"Priority: {safe_get(step,'priority')}")
                st.divider()
        else:
            st.info("No plan available.")

        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['PLAN'])}s")


    # ---------------- SEARCH ----------------
    with tab3:
        st.subheader("Search Results")

        search_results = result.get("search_results")

        if not search_results:
            st.error("No results found.")
        else:
            for step_id, results in search_results.items():
                with st.expander(f"Step {step_id}"):
                    for r in results:
                        st.markdown(f"**{safe_get(r,'title','Untitled')}**")

                        snippet = safe_get(r, "snippet")
                        if snippet:
                            st.write(snippet)

                        st.caption(safe_get(r, "url", ""))
                        st.caption(f"Quality: {round(safe_get(r,'quality_score',0),2)}")

                        st.divider()

        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['SEARCH'])}s")


    # ---------------- EVALUATION ----------------
    with tab4:
        st.subheader("Evaluation")

        evaluation = result.get("evaluation")

        if evaluation:
            st.markdown(f"**Decision:** `{safe_get(evaluation,'decision')}`")
            st.markdown(f"**Confidence:** `{round(result.get('overall_confidence',0),3)}`")

            for step in safe_get(evaluation, "steps", []):
                st.write({
                    "step": safe_get(step, "step_id"),
                    "confidence": round(safe_get(step, "confidence_score", 0), 3),
                    "passed": safe_get(step, "passed")
                })
        else:
            st.info("No evaluation data available.")

        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['EVALUATION'])}s")


    # ---------------- SYNTHESIS ----------------
    with tab5:
        st.subheader("Synthesised Claims")

        synthesis = result.get("synthesis")

        if synthesis and getattr(synthesis, "claims", None):
            for claim in synthesis.claims:
                st.markdown(f"- {claim.text}")

                citation_text = " ".join(claim.citation_ids) if claim.verified else "UNVERIFIED"

                st.caption(f"Confidence: {round(claim.confidence,2)} | Citations: {citation_text}")
                st.divider()
        else:
            st.info("No synthesis available.")

        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['SYNTHESIS'])}s")


    # ---------------- CITATIONS ----------------
    with tab6:
        st.subheader("Citations")

        report = result.get("report")

        if report and getattr(report, "citations", None):

            for c in report.citations:

                cid = c.get("id")
                title = c.get("title", "Untitled")
                url = c.get("url", "")
                quality = c.get("quality_score", None)

                st.markdown(f"**{cid} {title}**")
                st.caption(url)

                if quality is not None:
                    st.caption(f"Quality: {round(quality, 3)}")

                st.divider()

        else:
            st.info("No citations available.")

        st.caption(f"Time: {get_node_time(node_logs, NODE_KEYS['CITATION'])}s")


    # ---------------- DEBUG ----------------
    with tab7:
        st.subheader("Node Execution Details")

        if node_logs:
            for node, data in node_logs.items():
                with st.expander(f"{node} NODE"):

                    # REMOVE TRACE FROM DEBUG DISPLAY
                    debug_data = {k: v for k, v in data.items() if k != "_trace"}

                    if debug_data:
                        st.json(debug_data)
                    else:
                        st.info("No debug data available")

        else:
            st.info("No node logs available.")

        st.divider()

        errors = result.get("errors")

        if errors:
            for err in errors:
                st.error(str(err))
        else:
            st.success("No errors encountered.")