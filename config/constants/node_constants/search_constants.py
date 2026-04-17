MAX_SEARCH_RETRIES = 3
BACKOFF_BASE = 1

MAX_RESULTS_PER_STEP = 5
MAX_PER_DOMAIN = 2
MAX_SCRAPES = 3
TAVILY_MAX_RESULTS = 8

# Common English words excluded from title–query Jaccard overlap in pre-scrape scoring.
# Removing these prevents high-frequency function words from inflating topical similarity.
STOPWORDS = {
    "the", "a", "an", "is", "are", "in", "of", "for",
    "and", "to", "on", "at", "with", "by",
}