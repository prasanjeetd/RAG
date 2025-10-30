from typing import List, Tuple
from FlagEmbedding import Reranker

def bge_rerank(model_name: str, query: str, docs: List[str], top_n: int = 4) -> List[Tuple[int, float]]:
    rr = Reranker(model_name, use_fp16=True)
    pairs = [(query, d) for d in docs]
    scores = rr.compute_score(pairs, normalize=True)
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_n]
    return ranked
