from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. 加载文档
loader = TextLoader(
    "data/python.txt",
    encoding="utf-8"
)

documents = loader.load()


# 2. 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)


# 3. 创建 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. 创建 FAISS Vector Store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector store created successfully.")

query = "What is Python used for?"

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

results = retriever.invoke(query)

print("\nQuery:")
print(query)

print("\nRetrieved documents:")

for i, doc in enumerate(results):
    print(f"\nDocument {i}:")
    print(doc.page_content)
    print("Metadata:", doc.metadata)