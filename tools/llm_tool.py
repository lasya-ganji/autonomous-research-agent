import os
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langsmith import LangSmithTracer

# Load environment variables
load_dotenv()

# Create a global LangSmith tracer
tracer = LangSmithTracer()

def get_llm(model_name="gpt-4o-mini", temperature=0.2):
    """
    Returns a ChatOpenAI instance with LangSmith tracing enabled
    """
    return ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        callbacks=[tracer]
    )

def call_llm(prompt: str, llm=None, temperature: float = 0.2):
    """
    Call LLM with structured output and LangSmith tracing
    """
    if llm is None:
        llm = get_llm(temperature=temperature)

    # Create messages in the same structured format you had
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

    # LangChain expects input differently
    response = llm.generate([messages])

    # Extract text
    content = response.generations[0][0].text

    print("\n[LLM RAW OUTPUT]:\n", content)

    return content