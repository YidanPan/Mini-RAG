from loader import load_documents
from splitter import split_documents
from embedding import get_embeddings
from vectorstore import create_vectorstore
from vectorstore import save_vectorstore

#创建库程序
def build_index():
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")
    print("Splitting documents...")

    chunks = split_documents(documents)#切分document中的文本为chunk
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")
    embeddings = get_embeddings()#得到向量工具

    print("Building FAISS vector store...")
    vectorstore = create_vectorstore(
        chunks,
        embeddings
    )

    print("Saving vector store...")
    save_vectorstore(vectorstore)
    
    print("Vector store built successfully.")

if __name__ == "__main__":
    build_index()