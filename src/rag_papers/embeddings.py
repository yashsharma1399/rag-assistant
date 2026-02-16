# embeddings.py
# Handles embedding text using a local SentenceTransformer model.

from typing import List
from sentence_transformers import SentenceTransformer

# Loads a local embedding model.
# - all-MiniLM-L6-v2 is fast and good enough for a beginner RAG project.
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

# Converts list of strings into list of embedding vectors.
# We return Python lists (not numpy arrays) because Chroma accepts them easily.
# model: SentenceTransformer → expects an already-loaded model
# texts: List[str] → expects a list of strings (chunks)
def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    # converts each string into an embedding vector
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()

