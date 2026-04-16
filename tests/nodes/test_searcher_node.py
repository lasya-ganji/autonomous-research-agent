import pytest
from models.state import ResearchState
from nodes.searcher_node import searcher_node


class MockResult:
    def __init__(self, url, title, snippet):
        self.url = url
        self.title = title
        self.snippet = snippet


# -----------------------------
# TEST 1: basic success flow
# -----------------------------
def test_searcher_generates_results():

    def mock_search(query):
        return [
            MockResult("https://example.com/1", "Test 1", "Snippet 1"),
            MockResult("https://example.com/2", "Test 2", "Snippet 2"),
        ]

    def mock_dedup(results, embedding_fn=None):
        return results

    def mock_validate(url):
        return "valid"

    def mock_scrape(url):
        return {"content": "Some content", "publish_date": None}

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "What is AI?", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = mock_dedup
    sn.validate_url = mock_validate
    sn.scrape_url = mock_scrape

    result = searcher_node(state)

    assert 1 in result.search_results
    assert len(result.search_results[1]) > 0
    assert len(result.citations) > 0


# -----------------------------
# TEST 2: fallback search works
# -----------------------------
def test_searcher_fallback_query():

    def mock_search(query):
        if "fallback" in query:
            return [MockResult("https://example.com", "Fallback", "")]
        return []

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "fallback test query", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = lambda x, embedding_fn=None: x
    sn.validate_url = lambda x: "valid"
    sn.scrape_url = lambda x: {}

    result = searcher_node(state)

    assert len(result.search_results[1]) > 0


# -----------------------------
# TEST 3: handles empty results
# -----------------------------
def test_searcher_no_results():

    def mock_search(query):
        return []

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "No results query", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = lambda x, embedding_fn=None: x
    sn.validate_url = lambda x: "valid"
    sn.scrape_url = lambda x: {}

    result = searcher_node(state)

    assert result.search_results[1] == []


# -----------------------------
# TEST 4: deduplication works
# -----------------------------
def test_searcher_deduplication():

    def mock_search(query):
        return [
            MockResult("https://same.com", "A", ""),
            MockResult("https://same.com", "B", ""),
        ]

    def mock_dedup(results, embedding_fn=None):
        return results  # let node handle URL dedup

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Dedup test", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = mock_dedup
    sn.validate_url = lambda x: "valid"
    sn.scrape_url = lambda x: {}

    result = searcher_node(state)

    assert len(result.search_results[1]) == 1


# -----------------------------
# TEST 5: error handling
# -----------------------------
def test_searcher_error_handling():

    def mock_search(query):
        raise Exception("Search failed")

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Error test", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search

    result = searcher_node(state)

    # Should not crash
    assert result is not None