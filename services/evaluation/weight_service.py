from typing import Dict
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
import json

def get_dynamic_weights(query: str) -> Dict[str, float]:
    # Load evaluator_weights.txt prompt
    prompt_template = load_prompt("evaluator_weights.txt")

    prompt = prompt_template.format(query=query)

    response = call_llm(prompt)

    try:
        weights = json.loads(response)

        # convert all numeric values to float (ignore 'reason' key)
        numeric_weights = {}
        for k, v in weights.items():
            if k in ["relevance", "recency", "domain", "depth"]:
                numeric_weights[k] = float(v)
        return numeric_weights
    except Exception:
        # fallback in case of parsing error
        return {
            "relevance": 0.5,
            "recency": 0.2,
            "domain": 0.2,
            "depth": 0.1
        }