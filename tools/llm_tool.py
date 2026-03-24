import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5)
)
def call_llm(
    prompt: str,
    temperature: float = 0.2,
    expect_json: bool = True
):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            temperature=temperature,
            response_format={"type": "json_object"} if expect_json else None,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise and reliable AI assistant. "
                        "Follow instructions strictly."
                        "If JSON is required, return ONLY valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            timeout=30 \
        )

        content = response.choices[0].message.content

        # JSON MODE
        if expect_json:
            parsed = safe_json_parse(content)

            if parsed is None:
                
                return {
                    "error": "invalid_json",
                    "raw_output": content
                }

            return parsed

        # TEXT MODE
        return content

    except Exception as e:
        #return structured error only
        return {
            "error": "llm_failure",
            "message": str(e)
        }