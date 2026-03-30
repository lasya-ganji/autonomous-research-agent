import streamlit as st
from agent_runner import run_agent

# Page Config
st.set_page_config(
    page_title="Autonomous Research Agent",
    layout="wide"
)

# Header
st.title("Autonomous Research Agent")
st.markdown(
    "Enter a query and get a structured, citation-backed research report."
)

# Input Section
query = st.text_input(
    "Research Query",
    placeholder="Enter your query here..."
)

run_button = st.button("Run Research")

# Run Agent
if run_button:

    if not query.strip():
        st.warning("Please enter a valid query.")
        st.stop()

    with st.spinner("Running research agent..."):
        result = run_agent(query)

    
    try:
        result = result.dict()
    except Exception:
        pass

    st.success("Research completed successfully!")

    # TABS UI 

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Report",
        "Plan",
        "Search",
        "Evaluation",
        "Synthesis",
        "Citations", 
        "Debug"
    ])


    # TAB 1 — REPORT
    with tab1:
        st.subheader("Final Report")

        report = result.get("report")

        if report and report.sections:
            st.markdown(report.sections[0])
        else:
            st.error("No report generated.")

    # TAB 2 — PLAN
    with tab2:
        st.subheader("Research Plan")

        plan = result.get("research_plan")

        if plan:
            for step in plan:
                st.markdown(f"**Step {step.step_id}**")
                st.write(step.question)
                st.caption(f"Priority: {step.priority}")
        else:
            st.info("No plan available.")

    # TAB 3 — SEARCH
    with tab3:
        st.subheader("Search Results")

        search_results = result.get("search_results")

        if search_results:
            for step_id, results in search_results.items():
                with st.expander(f"Step {step_id} Results"):
                    for r in results:
                        st.markdown(f"**{r.title}**")
                        st.write(r.snippet)
                        st.caption(r.url)
                        st.divider()
        else:
            st.info("No search results.")

    # TAB 4 — EVALUATION
    with tab4:
        st.subheader("Evaluation")

        evaluation = result.get("evaluation")

        if evaluation:
            st.write(f"**Decision:** `{evaluation.decision}`")

            for step in evaluation.steps:
                st.write({
                "step_id": step.step_id,
                "confidence": round(step.confidence_score, 3),
                "passed": step.passed
            })
        else:
            st.info("No evaluation data available.")

    # TAB 5 — SYNTHESIS
    with tab5:
        st.subheader("Synthesised Claims")

        synthesis = result.get("synthesis")

        if synthesis and synthesis.claims:
            for claim in synthesis.claims:
                st.markdown(f"- {claim.text}")
                st.caption(
                    f"Confidence: {round(claim.confidence, 2)} | "
                    f"Citations: {claim.citation_ids}"
                )

            if synthesis.partial:
                st.warning("Partial synthesis generated due to limited data.")
        else:
            st.info("No synthesis available.")

    # TAB 6 — CITATIONS
    with tab6:
        st.subheader("Citations")

        report = result.get("report")
        all_citations = result.get("citations")

        if report and report.citations and all_citations:

            used_ids = set(report.citations)

            # 🔥 Filter only used citations
            filtered_citations = [
                c for cid, c in all_citations.items()
                if cid in used_ids and c.status == "valid"
            ]

            for i, c in enumerate(filtered_citations):

                # Status color
                if c.status == "valid":
                    status_color = "🟢"
                elif c.status == "stale":
                    status_color = "🟡"
                else:
                    status_color = "🔴"

                # 🔥 Show index number matching [1], [2]
                st.markdown(f"### [{i+1}] {status_color} {c.title}")

                st.write(f"**Citation ID:** {c.citation_id}")
                st.write(f"**URL:** {c.url}")
                st.write(f"**Quality Score:** {round(c.quality_score, 2)}")
                st.write(f"**Status:** `{c.status}`")
                st.caption(f"Accessed: {c.date_accessed}")

                st.divider()

        else:
            st.info("No citations available.")

    # TAB 7 — DEBUG
    with tab7:
        st.subheader("Node Execution Details")

        # Node Logs
        node_logs = result.get("node_logs", {})

        if node_logs:
            for node, data in node_logs.items():
                with st.expander(f"{node.upper()} NODE"):
                    st.json(data)
        else:
            st.info("No node logs available.")

        st.divider()

        # Errors
        st.subheader("Errors (if any)")

        errors = result.get("errors")

        if errors:
            for err in errors:
                try:
                    st.error(f"[{err.severity}] {err.node} → {err.message}")
                except Exception:
                    st.error(str(err))
        else:
            st.success("No errors encountered.")