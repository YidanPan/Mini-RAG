from config import RETRIEVAL_K

# 从存储的向量中检索
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )