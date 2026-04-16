from config.constants.chunking_constants import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP):
    
    chunks = []

    if not text:
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks