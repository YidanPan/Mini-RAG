"""Terminal entry point for Agentic RAG."""
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from agent import create_rag_agent
from agent_stream import stream_agent
from cli_commands import handle_command, is_exit_command, print_help
from embedding import get_embeddings
from llm import get_llm
from startup_check import run_startup_check
from vectorstore import load_vectorstore


def main():
    if not run_startup_check():
        return

    print("\nLoading Agentic RAG...")
    vectorstore = load_vectorstore(get_embeddings())
    runtime_options = {"web_search_enabled": True}
    agent = create_rag_agent(
        llm=get_llm(),
        vectorstore=vectorstore,
        checkpointer=InMemorySaver(),
        runtime_options=runtime_options,
    )
    config = {"configurable": {"thread_id": "mini-rag-session"}}

    def clear_conversation():
        config["configurable"]["thread_id"] = f"mini-rag-session-{uuid4().hex}"

    print("Agent ready.")
    print_help("Agentic RAG")

    while True:
        query = input("\nYou: ").strip()
        if is_exit_command(query):
            break
        if not query:
            continue
        if handle_command(
            query,
            vectorstore,
            clear_conversation,
            "Agentic RAG",
            runtime_options,
        ):
            continue
        stream_agent(agent=agent, query=query, config=config)


if __name__ == "__main__":
    main()
