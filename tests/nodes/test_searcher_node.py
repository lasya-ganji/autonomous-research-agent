from nodes.searcher_node import searcher_node
from models.state import ResearchState

class MockSearchTool:
    def search(self, query):
        return [
            {"title": "Test", "url": "http://test.com", "snippet": "data"}
        ]

def test_searcher_returns_results():
    state = ResearchState(query="AI")

    result = searcher_node(state, search_tool=MockSearchTool())

    assert result.search_results is not None