from graph.build_graph import build_graph
from models.state import ResearchState
from observability.langsmith_config import setup_langsmith
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from config.constants.query_constants import CREATION_VERBS, REFRAME_STRIP_PREFIXES


def classify_query_intent(query: str) -> str:
    """
    Classifies query as RESEARCH or CREATE using LLM.
    Returns 'CREATE' if the user is asking to generate/produce something,
    'RESEARCH' for information/analysis queries. Fails safe to RESEARCH.
    """
    prompt_template = load_prompt("query_classifier.txt")
    prompt = prompt_template.format(query=query)

    try:
        result = call_llm(prompt=prompt, temperature=0.0)

        if result.get("error"):
            return "RESEARCH"

        content = result.get("content", "").strip().upper()
        if "CREATE" in content:
            return "CREATE"
        return "RESEARCH"

    except Exception:
        return "RESEARCH"


def reframe_query(query: str) -> str:
    """
    Converts a CREATE-intent query into a precise RESEARCH question that
    preserves the original topic, intent depth, and domain specificity.

    Uses a structured prompt (query_reframer.txt) that applies:
      - intent extraction
      - subject identification
      - capability alignment

    Falls back to the original query on any failure, empty output,
    or if the LLM ignores instructions and returns an action verb.
    """
    prompt_template = load_prompt("query_reframer.txt")
    prompt = prompt_template.format(query=query)

    try:
        result = call_llm(prompt=prompt, temperature=0.0)

        if result.get("error"):
            return query

        raw = result.get("content", "").strip()

        # Take only the first non-empty line — guard against multi-line responses.
        reframed = next(
            (line.strip() for line in raw.splitlines() if line.strip()),
            ""
        )

        # Strip known prefixes the model may add despite instructions.
        lower = reframed.lower()
        for prefix in REFRAME_STRIP_PREFIXES:
            if lower.startswith(prefix):
                reframed = reframed[len(prefix):].strip().lstrip(":").strip()
                lower = reframed.lower()
                break

        # Guard: if the LLM returned an action verb as the first word,
        # the reframing failed — fall back to the original query.
        first_word = lower.split()[0] if lower.split() else ""
        if first_word in CREATION_VERBS:
            return query

        # Ensure the output ends with a question mark.
        if reframed and not reframed.endswith("?"):
            reframed += "?"

        return reframed if reframed else query

    except Exception:
        return query


def run_agent(query: str):
    setup_langsmith() 
    state = ResearchState(query=query)
    graph = build_graph()
    
    graph_obj = graph.get_graph()
    png_bytes = graph_obj.draw_mermaid_png()

    with open("graph.png", "wb") as f:
         f.write(png_bytes)
        
    result = graph.invoke(state)
    return result