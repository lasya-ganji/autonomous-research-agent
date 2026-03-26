from models.state import ResearchState


def route_after_evaluator(state: ResearchState) -> str:
    """
    Routes execution after evaluator node based on decision.
    """

    decision = state.evaluation.decision

    if decision == "proceed":
        return "synthesiser"

    elif decision == "retry":
        return "searcher"

    elif decision == "replan":
        return "planner"

    # fallback safety
    return "synthesiser"