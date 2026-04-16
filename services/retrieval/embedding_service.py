from typing import List, Optional
from sentence_transformers import SentenceTransformer
from config.constants.llm_constants import EMBEDDING_MODEL, EMBEDDING_PROMPT_PREFIX

# Load model once
model = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> Optional[List[float]]:
    if not text:
        print("[EMBEDDING] Empty text received")
        return None

    try:
        formatted_text = f"{EMBEDDING_PROMPT_PREFIX}{text}"

        embedding = model.encode(formatted_text)

        if embedding is None:
            print("[EMBEDDING] Model returned None")
            return None

        return embedding.tolist()

    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return None