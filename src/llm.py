from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def get_llm(model: str, base_url: str = "http://localhost:11434") -> ChatOllama:
    return ChatOllama(model=model, base_url=base_url, temperature=0.2)

def call_llm(llm: ChatOllama, system_prompt: str, user_prompt: str) -> str:
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = llm.invoke(msgs)
    return resp.content if hasattr(resp, "content") else str(resp)
