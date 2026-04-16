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

# Quality score mixing
RELEVANCE_MIX = 0.7
SECONDARY_MIX = 0.3
QUALITY_SCORE_MIN = 0.3
QUALITY_SCORE_FILTER = 0.25

# Depth word count thresholds
DEPTH_HIGH_WORDS = 1200
DEPTH_MED_WORDS = 700
DEPTH_LOW_WORDS = 300
DEPTH_HIGH_SCORE = 1.0
DEPTH_MED_SCORE = 0.75
DEPTH_LOW_SCORE = 0.55
DEPTH_MIN_SCORE = 0.3

# Recency weight tier boundaries
RECENCY_HIGH_WEIGHT = 0.3
RECENCY_MED_WEIGHT = 0.25
RECENCY_LOW_WEIGHT = 0.2
RECENCY_HIGH_DAYS = 365
RECENCY_MED_DAYS = 730
RECENCY_LOW_DAYS = 1200
RECENCY_STATIC_DAYS = 2000
RECENCY_MIN_SCORE = 0.1

# Outdated content threshold
OUTDATED_YEAR_THRESHOLD = 3

# Relevance boost/penalty adjustments
RELEVANCE_DUAL_BOOST = 0.05
RELEVANCE_TITLE_BOOST = 0.03
RELEVANCE_KEYWORD_PENALTY = 0.05

# Weight validation cap
WEIGHT_MAX_CAP = 0.6

# Embedding input truncation
DOC_EMBED_CHAR_LIMIT = 1000

# Recency fallback neutral score
RECENCY_NEUTRAL_SCORE = 0.5

# Recency year math
RECENCY_MAX_YEAR_DIFF = 20
RECENCY_DAYS_PER_YEAR = 365
RECENCY_MAX_DAYS_OLD = 3650

# Domain tier scores
DOMAIN_SCORE_HIGH_AUTHORITY = 0.95
DOMAIN_SCORE_RESEARCH = 0.9
DOMAIN_SCORE_GOV_EDU = 0.93
DOMAIN_SCORE_TRUSTED = 0.85
DOMAIN_SCORE_NEWS = 0.8
DOMAIN_SCORE_TECH_BLOG = 0.7
DOMAIN_SCORE_LOW_QUALITY = 0.5
DOMAIN_SCORE_BLOG_SUBDOMAIN = 0.65
DOMAIN_SCORE_DEFAULT = 0.75
DOMAIN_BLOG_SUBSTRINGS = ["blog", "dev", "tech"]

# Domain lists
HIGH_AUTHORITY_DOMAINS = [
    "ieee.org", "acm.org", "nature.com", "sciencedirect.com",
    "springer.com", "mit.edu", "stanford.edu", "harvard.edu",
    "nasa.gov", "who.int", "oecd.org", "worldbank.org"
]
RESEARCH_PLATFORM_DOMAINS = [
    "arxiv.org", "researchgate.net", "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov"
]
TRUSTED_KNOWLEDGE_DOMAINS = [
    "wikipedia.org", "britannica.com"
]
TECH_BLOG_DOMAINS = [
    "medium.com", "towardsdatascience.com", "substack.com", "hashnode.dev"
]
NEWS_DOMAINS = [
    "bbc.com", "nytimes.com", "reuters.com", "theguardian.com"
]
LOW_QUALITY_DOMAINS = [
    "quora.com", "reddit.com", "pinterest.com"
]