MIN_CONTENT_WORDS = 100
MAX_CONTENT_CHARS = 5000
SCRAPE_TIMEOUT = 8

# Minimum Tavily ML relevance score required before raw_content is accepted.
# Results below this threshold may have large word counts but are off-topic
# false positives — accepting their content inflates depth_score and pulls
# step confidence below threshold without adding useful information.
MIN_TAVILY_CONTENT_SCORE = 0.35

# Minimum Tavily score to include a result in the pipeline at all.
# Results below this are clearly off-topic (e.g. retirees pages at score 0.09)
# and should be dropped before deduplication, scoring, or scraping.
MIN_RESULT_SCORE = 0.20

# Minimum pre-scrape composite score required before attempting local scraping.
# URLs below this threshold get Tavily raw_content only (if available), or
# fall back to snippet. This prevents wasting scrape slots on borderline-
# relevance pages that would fill state with off-topic content.
SCRAPE_MIN_RELEVANCE = 0.40

# Domains that return no usable text content — filtered before scraping
BLOCKED_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "instagram.com",
    "linkedin.com",
    "facebook.com",
    "tiktok.com",
}

# Domains that frequently block scrapers — still attempted but deprioritised
DEPRIORITIZED_DOMAINS = {
    "medium.com",
}