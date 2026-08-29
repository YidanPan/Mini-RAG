from config import RETRIEVAL_K

# 从存储的向量中检索 通过mmr方式检索，提高相关性
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10 #从候选10个中选出3个
        }
    )