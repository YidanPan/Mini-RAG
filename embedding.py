from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

# 创建向量化模型
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )