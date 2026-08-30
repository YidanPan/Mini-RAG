from langchain.tools import tool
from retriever import get_retriever
from langgraph.config import (get_stream_writer)
from pathlib import Path

def format_documents(docs): #把document中的内容格式统一
    parts = []

    for doc in docs:
        source = doc.metadata.get(
            "source",
            "unknown"
        )
        page = doc.metadata.get("page")
        if page is not None:
            source_info = (
                f"{source}, page {page + 1}"
            )
        else:
            source_info = source
        parts.append(
            f"[Source: {source_info}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def create_tools(vectorstore):
    source = Path(source).name

    @tool
    def search_knowledge_base(query: str) -> str:
        """
        Search the entire local document knowledge base.

        Use this tool when the user asks about
        information that may exist in the local
        PDF, Markdown, or text documents.

        The query should be a clear standalone
        search query.
        """
        writer=get_stream_writer() #获取一个“往 Streaming 通道里发消息”的对象
        writer("Searching local knowledge base...")

        retriever = get_retriever(vectorstore)

        docs = retriever.invoke(query)#retrieve函数返回的结果都是document元素的list
        if not docs:
            return (
                "No relevant documents were found "
                "in the knowledge base."
            )

        writer(f"Retrieved {len(docs)} documents.")
        return format_documents(docs)

    @tool
    def search_document(query: str,source: str) -> str:
        """
        Search only a specific document in the
        local knowledge base.

        Use this tool when the user explicitly
        asks to search, answer from, or restrict
        the answer to a particular file.

        source must be the exact normalized file
        name, such as 'rag.pdf' or
        'transformer.pdf'.
        """
        writer=get_stream_writer()
        writer("Searching local knowledge base...")

        retriever = get_retriever(vectorstore,source=source)

        docs = retriever.invoke(query)
        if not docs:
            return (
                f"No relevant content was found "
                f"in {source}."
            )
        
        writer(f"Retrieved {len(docs)} documents.")
        return format_documents(docs)


    @tool
    def calculator(a: float,b: float,operation: str) -> str:
        """
        Perform basic arithmetic.

        operation must be one of:
        add, subtract, multiply, divide.
        """

        if operation == "add":
            result = a + b

        elif operation == "subtract":
            result = a - b

        elif operation == "multiply":
            result = a * b

        elif operation == "divide":
            if b == 0:
                return "Cannot divide by zero."
            result = a / b

        else:
            return (
                "Invalid operation. "
                "Use add, subtract, "
                "multiply, or divide."
            )
        return str(result)
    
    @tool
    def list_documents() -> str: #能读取 FAISS docstore 里的 metadata,返回所有source
        """
        List the document sources available in
        the local knowledge base.
        """

        sources = set()
        for doc in (
            vectorstore.docstore._dict.values()
        ):
            source = doc.metadata.get("source")
            if source:
                sources.add(source)
        return "\n".join(
            sorted(sources)
        )
    
    return [
        search_knowledge_base,
        search_document,
        calculator,
        list_documents
    ]