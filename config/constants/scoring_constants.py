import json
from pathlib import Path

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

# DOMAIN TIER SCORES 
DOMAIN_SCORE_HIGH_AUTHORITY = 0.95
DOMAIN_SCORE_RESEARCH       = 0.90
DOMAIN_SCORE_GOV_EDU        = 0.93  
DOMAIN_SCORE_TRUSTED        = 0.85
DOMAIN_SCORE_NEWS           = 0.80
DOMAIN_SCORE_TECH_BLOG      = 0.70
DOMAIN_SCORE_LOW_QUALITY    = 0.50
DOMAIN_SCORE_BLOG_SUBDOMAIN = 0.65
DOMAIN_SCORE_DEFAULT        = 0.75
DOMAIN_BLOG_SUBSTRINGS      = ["blog", "dev", "tech"]


# Academic TLDs used by universities and research institutes worldwide.
ACADEMIC_TLDS = (
    ".edu",       # US universities
    ".ac.uk",     # UK academic institutions
    ".ac.jp",     # Japanese universities
    ".ac.in",     # Indian universities
    ".ac.nz",     # New Zealand universities
    ".ac.za",     # South African universities
    ".ac.kr",     # Korean universities
    ".ac.il",     # Israeli universities
    ".edu.au",    # Australian universities
    ".edu.cn",    # Chinese universities
)

# Government TLDs — official public-sector sources worldwide
GOV_TLDS = (
    ".gov",       # US federal government
    ".gov.uk",    # UK government
    ".gov.au",    # Australian government
    ".gc.ca",     # Canadian government
    ".gov.nz",    # New Zealand government
    ".gov.in",    # Indian government
    ".gob.mx",    # Mexican government
    ".gouv.fr",   # French government
    ".gov.sg",    # Singapore government
)


# To add a new site: edit domain_authority.json and restart. No code change needed.
_DOMAIN_AUTHORITY_PATH = Path(__file__).parent.parent / "domain_authority.json"
try:
    with open(_DOMAIN_AUTHORITY_PATH, encoding="utf-8") as _f:
        _domain_data: dict = json.load(_f)
except FileNotFoundError:
    _domain_data = {
        "high_authority": [], "research_platforms": [],
        "trusted_knowledge": [], "news": [],
        "tech_blogs": [], "low_quality": [],
    }

# Backward-compatible flat exports (used by scoring_service.py and any other consumers)
HIGH_AUTHORITY_DOMAINS    = _domain_data.get("high_authority", [])
RESEARCH_PLATFORM_DOMAINS = _domain_data.get("research_platforms", [])
TRUSTED_KNOWLEDGE_DOMAINS = _domain_data.get("trusted_knowledge", [])
NEWS_DOMAINS              = _domain_data.get("news", [])
TECH_BLOG_DOMAINS         = _domain_data.get("tech_blogs", [])
LOW_QUALITY_DOMAINS       = _domain_data.get("low_quality", [])

# compute_domain() iterates this list so adding a new tier = one JSON entry + one dict here.
DOMAIN_TIERS = [
    {"name": "high_authority",    "domains": HIGH_AUTHORITY_DOMAINS,    "score": DOMAIN_SCORE_HIGH_AUTHORITY},
    {"name": "research",          "domains": RESEARCH_PLATFORM_DOMAINS, "score": DOMAIN_SCORE_RESEARCH},
    {"name": "trusted_knowledge", "domains": TRUSTED_KNOWLEDGE_DOMAINS, "score": DOMAIN_SCORE_TRUSTED},
    {"name": "news",              "domains": NEWS_DOMAINS,               "score": DOMAIN_SCORE_NEWS},
    {"name": "tech_blogs",        "domains": TECH_BLOG_DOMAINS,          "score": DOMAIN_SCORE_TECH_BLOG},
    {"name": "low_quality",       "domains": LOW_QUALITY_DOMAINS,        "score": DOMAIN_SCORE_LOW_QUALITY},
]