from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 加载所有 txt 文件
loader = DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8"
    }
)

documents = loader.load()
print("Number of documents:", len(documents))

# 切分文档
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)
print("Number of chunks:", len(chunks))

# 查看 chunks
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}:")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)

# 创建 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建 FAISS
vectorstore = FAISS.from_documents(chunks,embeddings)

# 保存到本地
vectorstore.save_local("faiss_index")
print("\nVector store saved successfully.")