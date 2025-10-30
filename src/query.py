import argparse
from src.config import *
from src.embedder import get_embedder
from src.vectorstore import get_qdrant
from src.reranker import bge_rerank
from src.utils import docs_to_strings
from src.prompts import STRICT_SYSTEM_PROMPT, build_user_prompt
from src.llm import get_llm, call_llm

def main(question: str):
    embedder = get_embedder(EMBED_MODEL)
    vs = get_qdrant(QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, embedder)

    # 1) Retrieve
    docs = vs.similarity_search(question, k=RETRIEVE_K)
    if not docs:
        print("No docs retrieved.")
        return

    # 2) Rerank (local BGE)
    doc_texts = [d.page_content for d in docs]
    try:
        reranked = bge_rerank(RERANKER_MODEL, question, doc_texts, top_n=RERANK_TOP_N)
        top_docs = [docs[i] for i, _ in reranked]
    except Exception as e:
        print(f"[WARN] Reranker failed: {e}. Falling back to vector top-k.")
        top_docs = docs[:RERANK_TOP_N]

    contexts = docs_to_strings(top_docs)
    user_prompt = build_user_prompt(question, contexts)

    # 3) Generate with Ollama
    llm = get_llm(OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    answer = call_llm(llm, STRICT_SYSTEM_PROMPT, user_prompt)

    print("\n==== Answer ====")
    print(answer)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="User question")
    args = ap.parse_args()
    main(args.q)
