from typing import Dict
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
import json


DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}


REQUIRED_KEYS = {"relevance", "recency", "domain", "depth"}


def _is_valid(weights: Dict[str, float]) -> bool:
    # check keys
    if not all(k in weights for k in REQUIRED_KEYS):
        return False

    # check range
    for v in weights.values():
        if not isinstance(v, (int, float)):
            return False
        if v < 0 or v > 1:
            return False

    # check sum ≈ 1
    total = sum(weights.values())
    if not (0.95 <= total <= 1.05):
        return False

    return True


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1
    return {k: v / total for k, v in weights.items()}


def get_dynamic_weights(query: str) -> Dict[str, float]:
    prompt_template = load_prompt("evaluator_weights.txt")
    prompt = prompt_template.replace("{query}", query)

    try:
        response = call_llm(prompt)

        data = json.loads(response)

        weights = {
            k: float(data[k])
            for k in REQUIRED_KEYS
            if k in data
        }

        if not _is_valid(weights):
            return DEFAULT_WEIGHTS

        weights = _normalize(weights)

        return weights

    except Exception:
        return DEFAULT_WEIGHTS