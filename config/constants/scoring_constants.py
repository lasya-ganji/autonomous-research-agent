DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}

MIN_THRESHOLDS = {
    "relevance": 0.35,
    "recency": 0.1,
    "domain": 0.1,
    "depth": 0.05
}

# relevance tuning (SAFE)
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.3
TITLE_WEIGHT = 0.1

SEMANTIC_BOOST_THRESHOLD = 0.6
KEYWORD_BOOST_THRESHOLD = 0.5
TITLE_BOOST_THRESHOLD = 0.4

MAX_TF_NORMALIZATION = 3