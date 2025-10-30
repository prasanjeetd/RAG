from typing import List
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_core.documents import Document

def get_qdrant(client_url: str, api_key: str | None, collection: str, embeddings):
    client = QdrantClient(url=client_url, api_key=api_key) if api_key else QdrantClient(url=client_url)
    vs = Qdrant(client=client, collection_name=collection, embeddings=embeddings)
    return vs

def upsert_chunks(vs, docs: List[Document]):
    vs.add_documents(docs)
