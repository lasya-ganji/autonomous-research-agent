from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> Optional[List[float]]:
    if not text:
        print("[EMBEDDING] Empty text received")
        return None

    try:
        formatted_text = f"Represent this sentence for retrieval: {text}"

        embedding = model.encode(formatted_text)

        if embedding is None:
            print("[EMBEDDING] Model returned None")
            return None

        return embedding.tolist()

    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return None