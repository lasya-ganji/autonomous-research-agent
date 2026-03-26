import streamlit as st
from app.agent_runner import run_agent

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
    placeholder=" "
)

run_button = st.button("Run Research")

# Run Agent
if run_button:

    if not query.strip():
        st.warning("Please enter a valid query.")
        st.stop()

    with st.spinner("Running research agent..."):
        result = run_agent(query)

    st.success("Research completed successfully!")

    # Final Report (MAIN OUTPUT)
    st.subheader("Final Report")

    report = result.get("report")

    if report and report.sections:
        st.markdown(report.sections[0])
    else:
        st.error("No report generated.")

    # Evaluation (hidden by default)
    with st.expander("Evaluation Details"):
        evaluation = result.get("evaluation")

        if evaluation:
            st.write(f"Decision: `{evaluation.decision}`")

            for step in evaluation.steps:
                st.write({
                    "step_id": step.step_id,
                    "confidence": round(step.confidence_score, 3),
                    "passed": step.passed
                })
        else:
            st.info("No evaluation data available.")

    # Synthesis (hidden)
    with st.expander("Synthesised Claims"):
        synthesis = result.get("synthesis")

        if synthesis and synthesis.claims:
            for claim in synthesis.claims:
                st.markdown(f"- {claim.text}")
                st.caption(
                    f"Confidence: {round(claim.confidence, 2)} | "
                    f"Citations: {claim.citation_ids}"
                )

        # Show partial flag if applicable
            if synthesis.partial:
                st.warning("Partial synthesis generated due to limited data.")
        else:
            st.info("No synthesis available.")


    # Errors (hidden)
    with st.expander("Errors (if any)"):
        errors = result.get("errors")

        if errors:
            for err in errors:
                try:
                    st.error(
                        f"[{err.severity}] {err.node} → {err.message}"
                    )
                except Exception:
                # fallback if error is still dict (safety)
                    st.error(str(err))
        else:
            st.success("No errors encountered.")

