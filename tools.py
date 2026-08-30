"""Tool definitions for the Desktop Agentic RAG entry point."""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.config import get_stream_writer
from retriever import get_retriever
from langchain_tavily import TavilySearch

load_dotenv(Path(__file__).resolve().with_name(".env"))#把API key放入环境变量中

#格式化tool调用错误，详细格式类型
def format_tool_error(error_type, tool_name, detail, source=None, recoverable=True,suggested_action=None):
    """Return a stable, model-readable error response for every tool."""
    lines = [f"ERROR_TYPE: {error_type}", f"TOOL: {tool_name}"]
    if suggested_action is not None:
        lines.append(
            f"SUGGESTED_ACTION: {suggested_action}"
        )
    if source is not None:
        lines.append(f"SOURCE: {source}")
    lines.extend(
        [
            f"DETAIL: {detail}",
            f"RECOVERABLE: {str(recoverable).lower()}",
        ]
    )
    return "\n".join(lines)


def available_sources(vectorstore):
    """Return normalized source names from the FAISS docstore."""
    return {
        Path(source).name
        for doc in vectorstore.docstore._dict.values()
        if (source := doc.metadata.get("source"))
    }


def write_progress(message):
    """Emit streaming progress when called inside a LangGraph run."""
    try:
        get_stream_writer()(message)
    except Exception:
        # Direct tool invocation, such as a unit test, has no stream writer.
        pass


def format_documents(docs):
    """Format retrieved documents together with their normalized sources."""
    parts = []
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        source_info = f"{source}, page {page + 1}" if isinstance(page, int) else source
        parts.append(f"[Source: {source_info}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_web_results(payload): #把web查询返回的错误内容（通常返回的是字典）格式化，如果返回的是其他格式就转换成字符串
    """Turn Tavily's structured response into readable evidence for the Agent."""
    if not isinstance(payload, dict):
        return str(payload)

    parts = []
    answer = payload.get("answer")
    if answer:
        parts.append(f"Summary: {answer}")

    for result in payload.get("results", []):
        title = result.get("title", "Untitled result")
        url = result.get("url", "")
        content = result.get("content", "")
        parts.append(f"[Web: {title}]\nURL: {url}\n{content}")
    return "\n\n".join(parts) or "Web search returned no readable results."

def create_tools(vectorstore, runtime_options=None):
    """Create Agent tools bound to the supplied local vector store."""
    runtime_options = runtime_options if runtime_options is not None else {}

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the entire local document knowledge base using a standalone query."""
        if not query.strip():
            return format_tool_error(
                "INVALID_ARGUMENT", "search_knowledge_base", "query must not be empty."
            )

        write_progress("Searching local knowledge base...")
        try:
            docs = get_retriever(vectorstore).invoke(query)
        except Exception as error:
            return format_tool_error(
                "RETRIEVAL_ERROR",
                "search_knowledge_base",
                f"{type(error).__name__}: {error}",
            )
        if not docs:
            return format_tool_error(
                "NO_RESULTS",
                "search_knowledge_base",
                "No relevant documents were found in the knowledge base.",
            )

        write_progress(f"Retrieved {len(docs)} documents.")
        return format_documents(docs)

    @tool
    def search_document(query: str, source: str) -> str:
        """Search one indexed document by exact filename and a standalone query."""
        source = Path(source).name.strip()
        if not source:
            return format_tool_error(
                "INVALID_ARGUMENT",
                "search_document",
                "source must be a non-empty filename.",
                source=source,
            )
        if not query.strip():
            return format_tool_error(
                "INVALID_ARGUMENT",
                "search_document",
                "query must not be empty.",
                source=source,
            )

        try:
            sources = available_sources(vectorstore)
        except Exception as error:
            return format_tool_error(
                "RETRIEVAL_ERROR",
                "search_document",
                f"Could not inspect indexed sources: {type(error).__name__}: {error}",
                source=source,
            )
        if source not in sources:
            return format_tool_error(
                "DOCUMENT_NOT_FOUND",
                "search_document",
                "The requested filename is not indexed. Use list_documents to see available files.",
                source=source,
                recoverable=True,
                suggested_action="list_documents"
            )

        write_progress("Searching local knowledge base...")
        try:
            docs = get_retriever(vectorstore, source=source).invoke(query)
        except Exception as error:
            return format_tool_error(
                "RETRIEVAL_ERROR",
                "search_document",
                f"{type(error).__name__}: {error}",
                source=source,
            )
        if not docs:
            return format_tool_error(
                "NO_RESULTS",
                "search_document",
                "The document exists, but no relevant content was found for this query.",
                source=source,
                recoverable=True,
                suggested_action="search_knowledge_base"
            )

        write_progress(f"Retrieved {len(docs)} documents.")
        return format_documents(docs)

    @tool
    def calculator(a: float, b: float, operation: str) -> str:
        """Perform add, subtract, multiply, or divide on two numbers."""
        if operation == "add":
            return str(a + b)
        if operation == "subtract":
            return str(a - b)
        if operation == "multiply":
            return str(a * b)
        if operation == "divide":
            if b == 0:
                return format_tool_error(
                    "INVALID_ARGUMENT", "calculator", "Cannot divide by zero.",recoverable=False
                )
            return str(a / b)
        return format_tool_error(
            "INVALID_ARGUMENT",
            "calculator",
            "operation must be add, subtract, multiply, or divide.",
        )

    @tool
    def list_documents() -> str:
        """List the document sources available in the local knowledge base."""
        try:
            sources = available_sources(vectorstore)
        except Exception as error:
            return format_tool_error(
                "RETRIEVAL_ERROR",
                "list_documents",
                f"Could not inspect indexed sources: {type(error).__name__}: {error}",
            )
        if not sources:
            return format_tool_error(
                "NO_RESULTS", "list_documents", "No document sources are indexed."
            )
        return "\n".join(sorted(sources))

    #新增网络查询功能
    @tool
    def web_search(query: str) -> str:
        """Search the public web for current, recent, or explicitly requested online information."""
        if not query.strip():
            return format_tool_error(
                "INVALID_ARGUMENT", "web_search", "query must not be empty."
            )
        if not runtime_options.get("web_search_enabled", True):
            return format_tool_error(
                "WEB_SEARCH_DISABLED",
                "web_search",
                "Web Search is disabled for this session. Use /web on to enable it.",
                recoverable=False,
            )
        if not os.getenv("TAVILY_API_KEY"):
            return format_tool_error(
                "CONFIGURATION_ERROR",
                "web_search",
                "TAVILY_API_KEY is missing. Add it to the project .env file.",
                recoverable=False,
            )

        write_progress("Searching the web...")
        try:
            result = TavilySearch(
                max_results=5,
                topic="general",
                search_depth="basic",
            ).invoke({"query": query})
        except Exception as error:
            return format_tool_error(
                "WEB_SEARCH_ERROR",
                "web_search",
                f"{type(error).__name__}: {error}",
            )

        return format_web_results(result)

    return [search_knowledge_base, search_document, calculator, list_documents, web_search]
