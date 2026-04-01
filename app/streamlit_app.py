import streamlit as st
from agent_runner import run_agent

st.set_page_config(
    page_title="Autonomous Research Agent",
    layout="wide"
)

st.title("Autonomous Research Agent")
st.markdown("Enter a query and get a structured, citation-backed research report.")

query = st.text_input("Research Query", placeholder="Enter your query here...")
run_button = st.button("Run Research")

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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Report", "Plan", "Search", "Evaluation", "Synthesis", "Citations", "Debug"
    ])

    # ---------------- REPORT ----------------
    with tab1:
        st.subheader("Final Report")

        report = result.get("report")

        if report:
            try:
                sections = report.sections
            except AttributeError:
                sections = report.get("sections")

            if sections:
                content = sections[0]

            st.markdown(
                f"<div style='font-size:16px; line-height:1.7'>{content}</div>",
                unsafe_allow_html=True
            )
        else:
            st.error("No report generated.")

    # ---------------- PLAN ----------------
    with tab2:
        st.subheader("Research Plan")

        plan = result.get("research_plan")

        if plan:
            for step in plan:
                st.markdown(f"**Step {step.step_id}**")
                st.write(step.question)
                st.caption(f"Priority: {step.priority}")
                st.divider()
        else:
            st.info("No plan available.")

    # ---------------- SEARCH ----------------
    with tab3:
        st.subheader("Search Results")

        search_results = result.get("search_results")

        if not search_results or all(len(v) == 0 for v in search_results.values()):
            st.error("Search failed or no results found.")
        else:
            for step_id, results in search_results.items():
                with st.expander(f"Step {step_id}", expanded=False):
                    for r in results:

                        st.markdown(f"**{r.title}**")

                        if r.snippet:
                            st.write(r.snippet)

                        st.caption(r.url)

                        st.markdown(
                            f"<small>Quality: {round(r.quality_score or 0,2)}</small>",
                            unsafe_allow_html=True
                        )

                        st.divider()

    # ---------------- EVALUATION ----------------
    with tab4:
        st.subheader("Evaluation")

        evaluation = result.get("evaluation")

        if evaluation:
            st.markdown(f"**Decision:** `{evaluation.decision}`")
            st.markdown(f"**Avg Confidence:** `{round(result.get('overall_confidence',0),3)}`")

            for step in evaluation.steps:
                st.write({
                    "step": step.step_id,
                    "confidence": round(step.confidence_score, 3),
                    "passed": step.passed
                })
        else:
            st.info("No evaluation data available.")

    # ---------------- SYNTHESIS ----------------
    with tab5:
        st.subheader("Synthesised Claims")

        synthesis = result.get("synthesis")

        if synthesis and synthesis.claims:
            for claim in synthesis.claims:

                st.markdown(f"• {claim.text}")

                st.caption(
                    f"Confidence: {round(claim.confidence,2)} | "
                    f"Citations: {' '.join(claim.citation_ids)}"
                )

                st.divider()

            if synthesis.partial:
                st.warning("Partial synthesis generated due to limited data.")
        else:
            st.info("No synthesis available.")

    # ---------------- CITATIONS ----------------
    with tab6:
        st.subheader("Citations")

        all_citations = result.get("citations")
        used_ids = result.get("used_citation_ids", [])

        if all_citations and used_ids:

            for cid in used_ids:
                c = all_citations.get(cid)

                if not c:
                    continue

                # status indicator
                if c.status == "valid":
                    status = "🟢"
                elif c.status == "stale":
                    status = "🟡"
                else:
                    status = "🔴"

                st.markdown(f"**{cid} {status} {c.title}**")
                st.caption(c.url)

                st.write(f"Quality: {round(c.quality_score,2)}")
                st.write(f"Status: `{c.status}`")

                st.divider()

        else:
            st.info("No citations available.")

    # ---------------- DEBUG ----------------
    with tab7:
        st.subheader("Node Execution Details")

        node_logs = result.get("node_logs", {})

        if node_logs:
            for node, data in node_logs.items():
                with st.expander(f"{node.upper()} NODE"):
                    st.json(data)
        else:
            st.info("No node logs available.")

        st.divider()

        st.subheader("Errors")

        errors = result.get("errors")

        if errors:
            for err in errors:
                try:
                    st.error(f"[{err.severity}] {err.node} → {err.message}")
                except Exception:
                    st.error(str(err))
        else:
            st.success("No errors encountered.")