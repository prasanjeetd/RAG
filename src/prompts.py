STRICT_SYSTEM_PROMPT = """You are a precise assistant.
Use ONLY the provided context to answer. If the answer is not in the context, say "I don't know."
Keep answers concise and include citations like [source]."""

def build_user_prompt(question: str, contexts_with_citations: list[str]) -> str:
    ctx_block = "\n\n".join(contexts_with_citations)
    return f"Question:\n{question}\n\nContext:\n{ctx_block}\n\nAnswer:"
