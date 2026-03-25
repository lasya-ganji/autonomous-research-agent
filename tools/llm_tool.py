import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str, temperature: float = 0.2, expect_json: bool = False):
    """
    Calls LLM and returns parsed response.

    Args:
        prompt (str): Input prompt
        temperature (float): Sampling temperature
        expect_json (bool): Whether JSON output is expected

    Returns:
        dict | list | str: Parsed response
    """

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
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
        )

        # Extract content
        content = response.choices[0].message.content

        print("\n[LLM RAW OUTPUT]:\n", content)

        if expect_json:
            try:
                content = re.sub(r"```json|```", "", content).strip()
                parsed = json.loads(content)

                # Handle case where model wraps list in object
                if isinstance(parsed, dict) and "steps" in parsed:
                    return parsed["steps"]

                return parsed

            except Exception:
                return {
                    "error": "Invalid JSON",
                    "raw": content
                }

        return content

    except Exception as e:
        return {
            "error": str(e)
        }