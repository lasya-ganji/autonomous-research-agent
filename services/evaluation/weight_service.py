from typing import Dict
from tools.llm_tool import call_llm
from utils.prompt_loader import load_prompt
from services.system.cost_tracker import calculate_cost   # ✅ ADD THIS
import json


DEFAULT_WEIGHTS = {
    "relevance": 0.5,
    "recency": 0.2,
    "domain": 0.2,
    "depth": 0.1
}

REQUIRED_KEYS = {"relevance", "recency", "domain", "depth"}


# ---------------- VALIDATION HELPERS ---------------- #

def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values()) or 1
    return {k: v / total for k, v in weights.items()}


def _apply_constraints(weights: Dict[str, float]) -> Dict[str, float]:
    weights["relevance"] = max(weights["relevance"], 0.35)
    weights["recency"] = max(weights["recency"], 0.15)
    weights["domain"] = max(weights["domain"], 0.15)
    weights["depth"] = max(weights["depth"], 0.05)

    weights = {k: min(v, 0.6) for k, v in weights.items()}

    return _normalize(weights)


def _is_valid(weights: Dict[str, float]) -> bool:
    if not weights or not all(k in weights for k in REQUIRED_KEYS):
        return False

    for v in weights.values():
        if not isinstance(v, (int, float)):
            return False
        if v < 0 or v > 1:
            return False

    return True


# ---------------- MAIN FUNCTION ---------------- #

def get_dynamic_weights(query: str, state=None) -> Dict[str, float]:
    prompt_template = load_prompt("evaluator_weights.txt")
    prompt = prompt_template.replace("{query}", query)

    try:
        # ✅ CALL LLM
        res = call_llm(prompt)

        response = res.get("content", "")
        usage = res.get("usage", {})

        # ---------------- TOKEN + COST TRACKING ---------------- #
        if state and usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Tokens
            state.total_tokens += usage.get("total_tokens", 0)

            # Cost
            cost = calculate_cost(prompt_tokens, completion_tokens)
            state.total_cost += cost

            # 🚨 COST GUARDRAIL
            if state.total_cost > state.cost_limit:
                state.abort = True
                return DEFAULT_WEIGHTS

        # ---------------- JSON PARSE ---------------- #
        try:
            data = json.loads(response)
        except Exception:
            return DEFAULT_WEIGHTS

        weights = {
            k: float(data.get(k, 0))
            for k in REQUIRED_KEYS
        }

        if not _is_valid(weights):
            return DEFAULT_WEIGHTS

        weights = _normalize(weights)
        weights = _apply_constraints(weights)

        return weights

    except Exception:
        return DEFAULT_WEIGHTS