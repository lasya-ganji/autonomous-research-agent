# Top-K documents used for confidence
TOP_K = 3

# Weights (must sum ~1.0)
QUALITY_WEIGHT = 0.4
COVERAGE_WEIGHT = 0.3
AGREEMENT_WEIGHT = 0.2
DIVERSITY_WEIGHT = 0.1

# Safety bounds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Embedding limits
MAX_TEXT_LENGTH = 1000