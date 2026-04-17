# Action verbs that must not appear at the start of a reframed query.
# If the LLM ignores reframing instructions and returns one of these as the
# first word, the output is rejected and the original query is used instead.
CREATION_VERBS = {
    "create", "generate", "build", "make", "write", "draft",
    "produce", "design", "construct", "implement", "develop",
}

# Prefixes the LLM sometimes prepends to its output despite being told not to.
# Stripped before the reframed query is returned.
REFRAME_STRIP_PREFIXES = [
    "output:", "reframed:", "research question:", "answer:",
    "result:", "transformed:", "here is", "here's",
    "i suggest", "suggested:",
]
