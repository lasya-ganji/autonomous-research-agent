from graph.build_graph import build_graph
from models.state import ResearchState
from observability.langsmith_config import setup_langsmith
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from services.system.cost_tracker import calculate_cost
from config.constants.query_constants import CREATION_VERBS, REFRAME_STRIP_PREFIXES
from langsmith import Client


def _extract_usage(result: dict) -> dict:
    """Extract token counts and cost from a call_llm result dict."""
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    cost = calculate_cost(prompt_tokens, completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost": cost,
    }


def classify_query_intent(query: str) -> tuple:
    """
    Classifies query as RESEARCH or CREATE using LLM.
    Returns (intent: str, meta: dict) where intent is 'CREATE' or 'RESEARCH'.
    Fails safe to RESEARCH on any error.
    """
    prompt_template = load_prompt("query_classifier.txt")
    prompt = prompt_template.format(query=query)

    try:
        result = call_llm(prompt=prompt, temperature=0.0)

        if result.get("error"):
            return "RESEARCH", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "fallback": True}

        meta = _extract_usage(result)
        meta["fallback"] = False

        content = result.get("content", "").strip().upper()
        intent = "CREATE" if "CREATE" in content else "RESEARCH"
        meta["intent"] = intent

        return intent, meta

    except Exception:
        return "RESEARCH", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "fallback": True}


def reframe_query(query: str) -> tuple:
    """
    Converts a CREATE-intent query into a precise RESEARCH question.
    Returns (reframed_query: str, meta: dict).
    Falls back to the original query on any failure.
    """
    prompt_template = load_prompt("query_reframer.txt")
    prompt = prompt_template.format(query=query)

    try:
        result = call_llm(prompt=prompt, temperature=0.0)

        if result.get("error"):
            return query, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "fallback": True}

        meta = _extract_usage(result)
        meta["fallback"] = False

        raw = result.get("content", "").strip()

        reframed = next(
            (line.strip() for line in raw.splitlines() if line.strip()),
            ""
        )

        lower = reframed.lower()
        for prefix in REFRAME_STRIP_PREFIXES:
            if lower.startswith(prefix):
                reframed = reframed[len(prefix):].strip().lstrip(":").strip()
                lower = reframed.lower()
                break

        first_word = lower.split()[0] if lower.split() else ""
        if first_word in CREATION_VERBS:
            meta["fallback"] = True
            return query, meta

        # Strip surrounding quotes and trailing punctuation before adding clean ?
        reframed = reframed.strip().strip('"').strip("'").rstrip("?").rstrip('"').rstrip("'").strip()

        if reframed:
            reframed += "?"

        final = reframed if reframed else query
        meta["reframed_query"] = final
        return final, meta

    except Exception:
        return query, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "fallback": True}


def run_agent(query: str):
    setup_langsmith() 
    state = ResearchState(query=query)
    graph = build_graph()
    
    graph_obj = graph.get_graph()
    png_bytes = graph_obj.draw_mermaid_png()

    with open("graph.png", "wb") as f:
         f.write(png_bytes)
        
    result = graph.invoke(state)
    try:
        Client().flush()
    except Exception:
        pass
    return result