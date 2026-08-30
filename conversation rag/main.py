"""Terminal entry point for Conversational RAG."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cli_commands import handle_command, is_exit_command, print_help
from embedding import get_embeddings
from llm import get_llm
from memory import ConversationMemory
from prompt import get_answer_prompt, get_rewrite_prompt
from rag import conversational_rag_stream, message_text
from retriever import get_retriever
from startup_check import run_startup_check
from vectorstore import load_vectorstore


def main():
    if not run_startup_check():
        return

    print("\nLoading Conversational RAG...")
    vectorstore = load_vectorstore(get_embeddings())

    source = input("Source filter (press Enter for all documents): ").strip() or None
    runtime_options = {"source_filter": source}
    memory = ConversationMemory()
    rewrite_prompt = get_rewrite_prompt()
    answer_prompt = get_answer_prompt()
    llm = get_llm()

    print_help("Conversational RAG")
    while True:
        query = input("\nQuestion: ").strip()
        if is_exit_command(query):
            break
        if not query:
            continue
        if handle_command(
            query,
            vectorstore,
            memory.clear,
            "Conversational RAG",
            runtime_options,
        ):
            continue

        retriever = get_retriever(
            vectorstore,
            source=runtime_options["source_filter"],
        )

        results = conversational_rag_stream(
            query=query,
            chat_history=memory.get_history(),
            retriever=retriever,
            rewrite_prompt=rewrite_prompt,
            answer_prompt=answer_prompt,
            llm=llm,
        )
        print("\nStandalone Question:")
        print(results["standalone_question"])
        print("\nAnswer:")
        full_answer = ""
        for chunk in results["stream"]:
            content = message_text(chunk.content)
            print(content, end="", flush=True)
            full_answer += content
        print("\n\nSources:")
        for source_name in results["sources"]:
            print(f"- {source_name}")

        memory.add_user_message(query)
        memory.add_ai_message(full_answer)


if __name__ == "__main__":
    main()
