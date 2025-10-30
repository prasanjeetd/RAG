from typing import Any
from sentence_transformers import SentenceTransformer

class STEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

def get_embedder(model_name: str) -> Any:
    return STEmbeddings(model_name)
