from langgraph.graph import StateGraph
from models.state import ResearchState

from graph.supervisor import supervisor_node
from nodes.planner_node import planner_node
from nodes.searcher_node import searcher_node
from nodes.evaluator_node import evaluator_node
from nodes.synthesiser_node import synthesiser_node
from nodes.citation_manager_node import citation_manager_node
from nodes.reporter_node import reporter_node


def route_from_supervisor(state: ResearchState) -> str:
    return state.next_node or "reporter"


def build_graph():
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("citation_manager", citation_manager_node)
    graph.add_node("reporter", reporter_node)

    # ENTRY POINT → supervisor
    graph.set_entry_point("supervisor")

    # Supervisor decides where to go
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "planner": "planner",
            "searcher": "searcher",
            "synthesiser": "synthesiser",
            "reporter": "reporter",
        },
    )

    # Core pipeline
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "evaluator")

    # After evaluator → ALWAYS go back to supervisor
    graph.add_edge("evaluator", "supervisor")

    # Final stages
    graph.add_edge("synthesiser", "citation_manager")
    graph.add_edge("citation_manager", "reporter")

    return graph.compile()