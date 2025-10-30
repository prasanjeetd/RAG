# RAG PoC (All Open-Source)
LangChain + Qdrant (Docker) + **Sentence-Transformers** embeddings + **BGE Reranker** (FlagEmbedding) + **Ollama** + Unstructured/pypdf + RAGAS

No OpenAI/Cohere keys required.

## Quick Start
1) Prereqs: Python 3.12, Docker, Ollama
2) `docker compose up -d` (starts Qdrant)
3) Create venv & install
   - Windows: `py -3.12 -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
   - Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
4) Pull an Ollama model: `ollama pull llama3.1`
5) Copy `.env.example` → `.env` (adjust model names if desired)
6) Ingest: `python -m src.ingest --data .\data`
7) Query: `python -m src.query --q "What is this project about?"`
8) Evaluate: `python -m src.eval_ragas --dataset .\eval\questions.csv`

## Default Models
- **Embeddings**: `BAAI/bge-small-en-v1.5` (CPU-friendly; for multilingual try `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` or `intfloat/multilingual-e5-small`)
- **Reranker**: `BAAI/bge-reranker-base` (via FlagEmbedding)
- **LLM**: `llama3.1` (via Ollama) — changeable in `.env`

## Notes
- First run will download Hugging Face models.
- You can change models in `.env` without code changes.
