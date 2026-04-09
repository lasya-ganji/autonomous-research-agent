from models.state import ResearchState


def route_after_evaluator(state: ResearchState) -> str:
    return state.next_node or "reporter"

    if decision == "proceed":
        return "proceed"

    elif decision == "retry":
        return "retry"

    elif decision == "replan":
        return "replan"

    return "proceed"