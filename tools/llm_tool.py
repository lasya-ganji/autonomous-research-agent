import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

#Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Safe JSON parser
def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

# Core LLM call
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
            model="gpt-4o",
            temperature=temperature,

            #Force JSON only when needed
            response_format={"type": "json_object"} if expect_json else None,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise and reliable AI assistant. "
                        "Always follow instructions strictly. "
                        "If JSON is required, return ONLY valid JSON."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        #Extract response text
        content = response.choices[0].message.content

        
        # JSON MODE (Evaluator / Planner)
       
        if expect_json:
            parsed = safe_json_parse(content)

            if parsed is None:
                #Instead of crashing → return safe fallback
                return {
                    "error": "invalid_json",
                    "raw_output": content,
                    "score": 0,
                    "confidence": "low"
                }

            return parsed

       
        # TEXT MODE (Synthesiser)
    
        return content

    except Exception as e:
        print(f"[LLM Error] {e}")

        # FINAL fallback (even after retries fail)
        if expect_json:
            return {
                "error": "llm_failure",
                "score": 0,
                "confidence": "low"
            }

        return "LLM failed to generate response."
