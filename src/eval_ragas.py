import argparse
import pandas as pd
from datasets import Dataset
from src.config import *
from src.embedder import get_embedder
from src.vectorstore import get_qdrant
from src.reranker import bge_rerank
from src.utils import docs_to_strings
from src.prompts import STRICT_SYSTEM_PROMPT, build_user_prompt
from src.llm import get_llm, call_llm

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy

def run_pipeline(question: str):
    embedder = get_embedder(EMBED_MODEL)
    vs = get_qdrant(QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, embedder)

    docs = vs.similarity_search(question, k=RETRIEVE_K)
    if not docs:
        return "", [], question

    doc_texts = [d.page_content for d in docs]
    try:
        reranked = bge_rerank(RERANKER_MODEL, question, doc_texts, top_n=RERANK_TOP_N)
        top_docs = [docs[i] for i, _ in reranked]
    except Exception:
        top_docs = docs[:RERANK_TOP_N]

    contexts = docs_to_strings(top_docs)
    user_prompt = build_user_prompt(question, contexts)

    llm = get_llm(OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    answer = call_llm(llm, STRICT_SYSTEM_PROMPT, user_prompt)
    return answer, [d.page_content for d in top_docs], question

def main(csv_path: str):
    df = pd.read_csv(csv_path)
    qs = df["question"].tolist()
    refs = df["reference"].tolist() if "reference" in df.columns else [None]*len(qs)

    records = []
    for q, ref in zip(qs, refs):
        answer, ctxs, _ = run_pipeline(q)
        records.append({
            "question": q,
            "answer": answer,
            "contexts": ctxs,
            "ground_truth": ref if ref is not None else ""
        })

    ds = Dataset.from_list(records)
    results = evaluate(
        ds,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
    )
    print("\n=== RAGAS Results ===")
    print(results)

    out_csv = "./eval/ragas_report.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with 'question' and optional 'reference' columns")
    args = ap.parse_args()
    main(args.dataset)
