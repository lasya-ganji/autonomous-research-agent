import os
from openai import OpenAI
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from config.constants.llm_constants import LLM_MODEL, DEFAULT_TEMPERATURE

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@traceable(name="llm_call")
def call_llm(prompt: str, temperature: float = DEFAULT_TEMPERATURE):
    try:
        messages = [
            {
                "role": "system",
                "content": """
You are a strict structured output generator.

Follow instructions EXACTLY.

If JSON is requested:
- Return ONLY valid JSON
- Do NOT include explanations, greetings, or extra text
- Ensure output is parseable using json.loads()
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=temperature
        )

        content = response.choices[0].message.content
        usage = response.usage

        run = get_current_run_tree()
        if run and usage:
            run.extra = {
                "token_usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }

        return {
            "content": content,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
        }

    except Exception as e:
        error_msg = str(e).lower()

        # -------------------------------
        # ERROR CLASSIFICATION
        # -------------------------------
        if any(x in error_msg for x in ["unauthorized", "invalid api key", "401"]):
            error_type = "api_error"

        elif any(x in error_msg for x in ["timeout", "timed out"]):
            error_type = "timeout_error"

        elif any(x in error_msg for x in ["connection", "network", "dns"]):
            error_type = "network_error"

        else:
            error_type = "unknown_error"

        return {
            "content": "",
            "usage": {},
            "error": str(e),
            "error_type": error_type,
            "error_source": "llm_call"
        }