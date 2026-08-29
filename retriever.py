from config import RETRIEVAL_K,RETRIEVAL_FETCH_K

# 从存储的向量中检索 通过mmr方式检索，提高相关性
def get_retriever(vectorstore, source=None):
    search_kwargs = {
        "k": RETRIEVAL_K,
        "fetch_k": RETRIEVAL_FETCH_K
    }
    if source:#如果有指定来源检索
        search_kwargs["filter"] = {
            "source": source
        }
    return vectorstore.as_retriever(
        search_type="mmr",#全库搜索按照mmr方式检索
        search_kwargs=search_kwargs
    )