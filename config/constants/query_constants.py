# Action verbs that must not appear at the start of a reframed query.
CREATION_VERBS = {
    "create", "generate", "build", "make", "write", "draft",
    "produce", "design", "construct", "implement", "develop",
}

# Stripped before the reframed query is returned.
REFRAME_STRIP_PREFIXES = [
    "output:", "reframed:", "research question:", "answer:",
    "result:", "transformed:", "here is", "here's",
    "i suggest", "suggested:",
]
