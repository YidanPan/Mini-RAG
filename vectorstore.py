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
        str(VECTORSTORE_PATH)
    )

def load_vectorstore(embeddings):
    index_file = VECTORSTORE_PATH / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTORSTORE_PATH}. Run build_index.py first."
        )
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore
