from graph.build_graph import build_graph
from models.state import ResearchState
from observability.langsmith_config import setup_langsmith
from tools.llm_tool import call_llm


def classify_query_intent(query: str) -> str:
    """
    Classifies query as RESEARCH or CREATE using LLM.
    Returns 'CREATE' if the user is asking to generate/produce something,
    'RESEARCH' for information/analysis queries.
    """
    prompt = f"""You are a query classifier for a research agent.

Classify the following query as either RESEARCH or CREATE.

RESEARCH: asking for information, explanation, comparison, analysis, history, overview, how something works, what something is
CREATE: asking the agent to generate, write, make, build, design, draft, produce a document, image, code, file, or template

Important: "how to create X" is RESEARCH. "Create X for me" is CREATE.

Query: "{query}"

Answer with one word only: RESEARCH or CREATE"""

    try:
        result = call_llm(prompt=prompt, temperature=0.0)
        content = result.get("content", "").strip().upper()
        if "CREATE" in content:
            return "CREATE"
        return "RESEARCH"
    except Exception:
        return "RESEARCH"


def reframe_query(query: str) -> str:
    """
    Converts a CREATE-intent query into a RESEARCH question
    that captures the same topic.
    """
    prompt = f"""Convert this creation request into a research question a search agent can answer.
Return ONLY the reframed research question. No explanation. No prefix. Just the question itself.

Original request: "{query}"
Reframed research question:"""

    try:
        result = call_llm(prompt=prompt, temperature=0.0)
        reframed = result.get("content", query).strip()

        for prefix in ["here is", "here's", "reframed:", "research question:", "i suggest", "suggested:"]:
            if reframed.lower().startswith(prefix):
                reframed = reframed[len(prefix):].strip().lstrip(":").strip()

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