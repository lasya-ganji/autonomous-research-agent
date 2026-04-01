import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str, temperature: float = 0.2):
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

        content = response.choices[0].message.content

        print("\n[LLM RAW OUTPUT]:\n", content)

        return content

    except Exception as e:
        return {
            "error": str(e)
        }