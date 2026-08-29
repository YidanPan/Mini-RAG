from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_PATH

#创建并加载FAISS
def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    return vectorstore

def save_vectorstore(vectorstore):
    vectorstore.save_local(
        VECTORSTORE_PATH
    )

def load_vectorstore(embeddings):
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore