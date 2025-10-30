import argparse
from src.config import *
from src.loaders import load_docs
from src.splitter import split_docs
from src.embedder import get_embedder
from src.vectorstore import get_qdrant, upsert_chunks

def main(data_dir: str):
    print("[INGEST] Loading docs from:", data_dir)
    raw_docs = load_docs(data_dir)
    print(f"[INGEST] Loaded {len(raw_docs)} raw docs")

    print("[INGEST] Splitting...")
    chunks = split_docs(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"[INGEST] Split into {len(chunks)} chunks")

    print("[INGEST] Embedding...")
    embedder = get_embedder(EMBED_MODEL)

    print("[INGEST] Connecting to Qdrant:", QDRANT_URL)
    vs = get_qdrant(QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, embedder)

    print("[INGEST] Upserting chunks...")
    upsert_chunks(vs, chunks)
    print("[INGEST] Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with source documents")
    args = ap.parse_args()
    main(args.data)
