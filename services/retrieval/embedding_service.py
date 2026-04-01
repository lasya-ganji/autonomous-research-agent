from typing import List
from sentence_transformers import SentenceTransformer

# Load model ONCE
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> List[float]:
    if not text:
        print(" Empty text for embedding")
        return None

    try:
        formatted_text = f"Represent this sentence for retrieval: {text}"

        embedding = model.encode(
            formatted_text
            
        )

        if embedding is None:
            print("Model returned None embedding")
            return None

        return embedding.tolist()

    except Exception as e:
        print(f"[EMBEDDING ERROR] {e}")
        return None