from dotenv import load_dotenv
import os
from services.retrieval.cache_service import SemanticCache

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

semantic_cache = SemanticCache(dim=384)