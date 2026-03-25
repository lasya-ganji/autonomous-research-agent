from graph.build_graph import build_graph
from models.state import ResearchState

def run_agent(query: str):
    state = ResearchState(query=query)
    graph = build_graph()
    
    graph_obj = graph.get_graph()
    # png_bytes = graph_obj.draw_mermaid_png()

    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)
        
    result = graph.invoke(state)
    return result