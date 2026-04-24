MAX_SEARCH_RETRIES = 3
BACKOFF_BASE = 1

MAX_RESULTS_PER_STEP = 5
MAX_PER_DOMAIN = 2
MAX_SCRAPES = 3
TAVILY_MAX_RESULTS = 8

# Pre-scrape ranking weights
PRE_SCRAPE_TAVILY_WEIGHT = 0.5
PRE_SCRAPE_SNIPPET_WEIGHT = 0.3
PRE_SCRAPE_TITLE_WEIGHT = 0.2
PRE_SCRAPE_TAVILY_DEFAULT = 0.5
SNIPPET_DENSITY_NORM = 60.0

# LLM curation
CURATOR_TEMPERATURE = 0.0
CURATOR_TIMEOUT = 30.0
CURATOR_SNIPPET_LIMIT = 300

# Common English words excluded from title–query Jaccard overlap in pre-scrape scoring.
STOPWORDS = {
    "the", "a", "an", "is", "are", "in", "of", "for",
    "and", "to", "on", "at", "with", "by",
}