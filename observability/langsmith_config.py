import os
from dotenv import load_dotenv


def setup_langsmith():
    load_dotenv()

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "autonomous-research-agent")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        print("LANGCHAIN_API_KEY not set")
    else:
        print("LangSmith API Key Loaded")

    print("LangSmith tracing enabled")