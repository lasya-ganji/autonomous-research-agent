import pytest
from models.state import ResearchState
from nodes.searcher_node import searcher_node

# 110 words of content — triggers Tier 0 auto-accept (>= 100 words + score >= 0.50)
# so the LLM curation call is skipped entirely in tests.
_MOCK_CONTENT = ("word " * 110).strip()


class MockResult:
    def __init__(self, url, title, snippet, content=None, relevance_score=0.8):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.content = content          # None = no Tavily pre-fetch
        self.relevance_score = relevance_score


# TEST 1: basic success flow
def test_searcher_generates_results():

    def mock_search(query, exclude_domains=None):
        return [
            MockResult(
                "https://example.com/1", "Test Article One",
                "This snippet has more than eight words total for the gate",
                content=_MOCK_CONTENT,
            ),
            MockResult(
                "https://example.com/2", "Test Article Two",
                "Another snippet with sufficient word count to pass gate three",
                content=_MOCK_CONTENT,
            ),
        ]

    def mock_dedup(results, embedding_fn=None):
        return results

    def mock_validate(url):
        return "valid"

    def mock_scrape(url):
        return {"content": "Some scraped content here", "publish_date": None, "status": "success"}

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

# TEST 2: fallback search works
def test_searcher_fallback_query():
    """
    Primary query (10 words) returns [].
    Fallback query (first 6 words) returns a result.
    """

    def mock_search(query, exclude_domains=None):
        if len(query.split()) <= 6:
            return [
                MockResult(
                    "https://example.com/fallback", "Fallback Result Title Here",
                    "This fallback snippet has more than eight words total here",
                    content=_MOCK_CONTENT,
                )
            ]
        return []

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {
            "step_id": 1,
            "question": "What is the exact fallback behavior search testing mechanism",
            "priority": 1,
        })
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = lambda x, embedding_fn=None: x
    sn.validate_url = lambda x: "valid"
    sn.scrape_url = lambda x: {"status": "success", "content": _MOCK_CONTENT, "publish_date": None}

    result = searcher_node(state)

    assert len(result.search_results[1]) > 0

# TEST 3: handles empty results
def test_searcher_no_results():

    def mock_search(query, exclude_domains=None):
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


# TEST 4: deduplication works
def test_searcher_deduplication():
    """
    Two results with the same URL — node's cross-step dedup keeps only the first.
    """

    def mock_search(query, exclude_domains=None):
        return [
            MockResult(
                "https://same.com/article", "Article A Title Here",
                "First snippet with enough words to pass the gate check here",
                content=_MOCK_CONTENT,
            ),
            MockResult(
                "https://same.com/article", "Article B Title Here",
                "Second snippet with enough words to pass the gate check here",
                content=_MOCK_CONTENT,
            ),
        ]

    def mock_dedup(results, embedding_fn=None):
        return results  

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Dedup test query here", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search
    sn.deduplicate_pipeline = mock_dedup
    sn.validate_url = lambda x: "valid"
    sn.scrape_url = lambda x: {"status": "success", "content": _MOCK_CONTENT, "publish_date": None}

    result = searcher_node(state)

    assert len(result.search_results[1]) == 1

# TEST 5: error handling
def test_searcher_error_handling():

    def mock_search(query, exclude_domains=None):
        raise Exception("Search failed")

    state = ResearchState(query="AI")
    state.research_plan = [
        type("Step", (), {"step_id": 1, "question": "Error test query here", "priority": 1})
    ]

    import nodes.searcher_node as sn
    sn.search_tool = mock_search

    result = searcher_node(state)

    # Should not crash
    assert result is not None
