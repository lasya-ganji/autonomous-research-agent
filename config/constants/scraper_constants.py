import re

MIN_CONTENT_WORDS = 100
MAX_CONTENT_CHARS = 5000
SCRAPE_TIMEOUT = 8

# Tavily content quality gates
MIN_TAVILY_CONTENT_SCORE = 0.35
MIN_RESULT_SCORE = 0.20

# Minimum pre-scrape relevance score to qualify for local scraping
SCRAPE_MIN_RELEVANCE = 0.40

# Relevance score above which a Tavily-pre-fetched source is auto-accepted
CURATE_AUTO_ACCEPT_SCORE = 0.50

# Snippet shorter than this indicates a login-wall or empty page
MIN_SNIPPET_WORDS = 8

# Skip a domain after this many scrape failures within one run
DOMAIN_FAIL_THRESHOLD = 2

# Binary / non-HTML file extensions.
BINARY_EXTENSIONS = re.compile(
    r'\.(?:pdf|docx?|pptx?|xlsx?|odt|odp|ods|'
    r'mp[34]|m4[av]|webm|avi|mov|mkv|flv|wmv|'
    r'wav|flac|ogg|aac|'
    r'zip|gz|tar|7z|rar|bz2|xz|'
    r'exe|dmg|pkg|msi|deb|rpm|'
    r'png|jpe?g|gif|webp|svg|ico|bmp|tiff?)'
    r'(?:\?[^/]*)?$',
    re.IGNORECASE,
)

# All patterns are path/structure based — no domain names.
NON_ARTICLE_URL_PATTERNS = re.compile(
    r'(?:'
    # Video player pages: /watch at URL end, /watch?, /watch/ID, /shorts/ID, /reels/ID
    r'/watch(?:$|\?|/[A-Za-z0-9_-])'
    r'|/shorts/[A-Za-z0-9_-]'
    r'|/reel[s]?/[A-Za-z0-9_-]'
    r'|/status/\d{15,}'
    r')',
    re.IGNORECASE,
)

# Hard-drop patterns for UGC / forum / thread URL structures.
UGC_DROP_URL_PATTERNS = re.compile(
    r'(?:'
    # Reddit: subreddit listing or thread page
    r'/r/[A-Za-z0-9_]{2,}/'
    # @username author paths — Medium personal posts, Instagram, Mastodon, Bluesky
    r'|/@[A-Za-z0-9_][A-Za-z0-9_.]{0,30}/'
    # Numeric Q&A question pages (Stack Overflow / Quora-style IDs)
    r'|/questions?/\d{4,}'
    # Forum thread / topic by numeric ID (vBulletin, phpBB, Discourse)
    r'|/threads?/\d{3,}'
    r'|/topic[s]?/\d{3,}'
    # Generic forum directory segment
    r'|/forums?/'
    # Discussion platform paths (Discourse, HN-style)
    r'|/discuss(?:ion[s]?)?/'
    r')',
    re.IGNORECASE,
)
