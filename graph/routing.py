from models.state import ResearchState


def route_after_evaluator(state: ResearchState) -> str:
    decision = state.evaluation.decision

    if decision == "proceed":
        return "proceed"

    elif decision == "retry":
        return "retry"

    elif decision == "replan":
        return "replan"

    return "proceed"