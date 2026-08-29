from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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


# 3. 创建 Embedding 模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. 把 Chunk 转换成向量
vectors = embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)


print("Number of chunks:", len(chunks))
print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))

query = "What is Python used for?"

query_vector = embeddings.embed_query(query)

print("\nQuery vector dimension:", len(query_vector))
print("First 5 values:", query_vector[:5])