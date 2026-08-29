from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# Load document
loader = TextLoader(
    "data/python.txt",
    encoding="utf-8"
)

documents = loader.load()

# Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

# Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Prompt
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

# LLM
llm = ChatOllama(
    model="qwen2.5:3b"
)

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )

rag_chain = (
    {  #一个用户的query同时传递给两个，context和question
        "context": retriever | format_docs, 
        "question": RunnablePassthrough() #原样输入
    }
    | prompt
    | llm
)

query = "What is Python used for?"

response = rag_chain.invoke(query)

print("\nAnswer:")
print(response.content)