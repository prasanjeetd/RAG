# Architecture (OSS)
```
User Query
  ↓
Sentence-Transformers Embedding (local)
  ↓
Qdrant vector search (k)
  ↓
BGE Reranker (local via FlagEmbedding)
  ↓
Prompt Builder (strict context)
  ↓
Ollama LLM (llama3.1/qwen2.5)
  ↓
Answer + Citations
```
