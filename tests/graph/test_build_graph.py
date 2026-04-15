from graph.build_graph import build_graph

def test_graph_build():
    graph = build_graph()
    compiled = graph.get_graph()

    assert compiled is not None