from langchain_ollama import ChatOllama
from config import LLM_MODEL

#应用LLM
def get_llm():
    return ChatOllama(
        model=LLM_MODEL
    )