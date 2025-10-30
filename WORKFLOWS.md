# Workflows (OSS)
- Ingest: Unstructured/pypdf → Split (1000/150) → Embedding (Sentence-Transformers) → Qdrant upsert
- Query: Embed → Qdrant (k=8) → BGE Reranker top-4 → Prompt → Ollama
- Eval: RAGAS metrics on questions.csv
