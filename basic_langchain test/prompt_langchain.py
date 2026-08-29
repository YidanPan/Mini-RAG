from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

#加载文档
loader = TextLoader(
    "data/python.txt",
    encoding="utf-8"
)

documents = loader.load()

#切分文本
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)

#Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector Store 储存向量和索引
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever 检索工具引入
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# 用户问题
query = "What is Python used for?"


# 检索
results = retriever.invoke(query)


# 把检索结果拼成 Context
context = "\n\n".join(
    doc.page_content
    for doc in results
)


# 创建 Prompt Template
prompt = PromptTemplate(
    template="""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

#生成最终 Prompt
final_prompt = prompt.invoke({
    "context": context,
    "question": query
})

#接入LLM
llm = ChatOllama(model="qwen2.5:3b")

response = llm.invoke(final_prompt) #把刚才生成的 Prompt 发送给 Qwen。

print("Final Prompt:")
print(final_prompt.text)

print(response.content)
