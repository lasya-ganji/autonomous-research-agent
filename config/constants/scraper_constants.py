import os
from dotenv import load_dotenv

# Load .env before any os.getenv() calls below.
# scraper_constants is imported at module level (before setup_langsmith or
# get_tavily_client run load_dotenv), so we must call it here to ensure
# SEARCH_EXCLUDE_DOMAINS from .env is read correctly.
load_dotenv()

MIN_CONTENT_WORDS = 100
MAX_CONTENT_CHARS = 5000
SCRAPE_TIMEOUT = 8

MIN_TAVILY_CONTENT_SCORE = 0.35

MIN_RESULT_SCORE = 0.20

SCRAPE_MIN_RELEVANCE = 0.40

# Pre-scrape signal gate: snippets shorter than this indicate empty/login-wall pages.
MIN_SNIPPET_WORDS = 8

# Structural gate: binary/non-HTML file extensions — never contain article text.
NON_HTML_EXTENSIONS = {
    ".pdf", ".mp4", ".mp3", ".avi", ".mov", ".mkv",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".zip", ".exe", ".dmg", ".tar", ".gz",
}

# Runtime learning: skip a domain after this many scrape failures in one run.
DOMAIN_FAIL_THRESHOLD = 2

# ---------------------------------------------------------------------------
# URL STRUCTURAL PATTERNS (content-type classification, no domain names)
# ---------------------------------------------------------------------------
# These patterns identify CONTENT TYPES by URL structure, not by domain name.
# A new video platform launched tomorrow that uses /watch?v= is caught
# automatically — without adding it to any list.
#
# Pattern rationale:
#   /watch(?:[?/]|$)   → video watch page (with OR without ?v= — catches bare /watch
#                         redirects and watch?v= links). (?:[?/]|$) ensures /watchdog
#                         and /watchman are NOT matched (word boundary via suffix).
#   /shorts/[id]       → short-form vertical video (YouTube Shorts-convention)
#   /reel/[id]         → video reel (Instagram/Facebook convention)
#   /video/\d{5,}      → video by long numeric ID (TikTok/Vimeo convention)
#   /status/\d{8,}     → microblog post by long numeric ID (Twitter/X convention)
#   /r/[name]/comments/ → Reddit-style hierarchical forum thread
UNSEARCHABLE_URL_PATTERNS = [
    r"/watch(?:[?/]|$)",              # video watch page — with or without ?v= param
    r"/shorts/[A-Za-z0-9_\-]+",     # short-form video
    r"/reel/[A-Za-z0-9_\-]+",       # video reel
    r"/video/\d{5,}",                # video by long numeric ID
    r"/status/\d{8,}",               # microblog post by long numeric ID
    r"/r/[A-Za-z0-9_]+/comments/",  # Reddit-style forum thread
]

# ---------------------------------------------------------------------------
# SEARCH-LEVEL DOMAIN EXCLUSIONS (minimal, environment-configurable)
# ---------------------------------------------------------------------------
# Only for platforms where the ENTIRE domain is structurally unusable and
# no URL pattern can detect it (auth-walled networks, image boards).
#
# This list is intentionally minimal — URL patterns above handle the rest.
# Override or extend without touching code:
#   set SEARCH_EXCLUDE_DOMAINS=linkedin.com,example.com in your .env
#
# youtu.be  — short YouTube links (path IS the video hash, e.g. youtu.be/abc123);
#             no URL pattern can distinguish this from a legitimate short hash URL
# quora.com — auth-walled Q&A (always returns 403); user-generated, non-authoritative,
#             non-citable; snippet-only with dom=0.500 even when it slips through
# instagram.com — photo posts at /p/ABCDEF/ have no generalizable pattern
# linkedin.com  — entire domain is auth-walled professional network
# facebook.com  — auth-walled social network; URL structure too varied for patterns
# pinterest.com — image board; no article text anywhere on domain
_exclude_defaults = (
    "youtu.be,quora.com,instagram.com,linkedin.com,facebook.com,pinterest.com,"
    "twitter.com,x.com,tiktok.com,snapchat.com,discord.com"
)
SEARCH_EXCLUDE_DOMAINS = [
    d.strip()
    for d in os.getenv("SEARCH_EXCLUDE_DOMAINS", _exclude_defaults).split(",")
    if d.strip()
]
