from typing import List
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
import pathlib

def load_docs(data_dir: str) -> List[Document]:
    docs: List[Document] = []
    p = pathlib.Path(data_dir)
    for path in p.glob("**/*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in [".pdf"]:
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            else:
                loader = UnstructuredFileLoader(str(path))
                docs.extend(loader.load())
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
    return docs
