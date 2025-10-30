from typing import List
from langchain_core.documents import Document

def docs_to_strings(docs: List[Document]) -> List[str]:
    out = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_number", ""))
        cite = f" [{src}:{page}]" if page != "" else f" [{src}]"
        out.append((d.page_content.strip(), cite))
    return [f"{text}{cite}" for text, cite in out]
