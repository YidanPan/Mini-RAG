"""Shared terminal commands for Mini RAG entry points."""

from config import LLM_MODEL, RETRIEVAL_FETCH_K, RETRIEVAL_K
from startup_check import run_startup_check


def is_exit_command(query):
    return query.lower() in {"exit", "quit", "/exit", "/quit"}


def _documents(vectorstore):
    return sorted(
        {
            document.metadata.get("source")
            for document in vectorstore.docstore._dict.values()
            if document.metadata.get("source")
        }
    )


def print_help(mode):
    print(f"\n{mode} commands:")
    print("  /help                 Show this help")
    print("  /status               Re-run the startup check")
    print("  /config               Show the active non-secret configuration")
    print("  /docs                 List indexed local documents")
    print("  /clear                Start a new conversation")
    if mode == "Conversational RAG":
        print("  /source <filename>    Restrict retrieval to one indexed document")
        print("  /source all           Search every indexed document")
    if mode == "Agentic RAG":
        print("  /web on|off           Enable or disable Web Search")
    print("  /exit                 Exit the program")
    print("\nAsk a normal question to continue chatting.")


def _print_config(mode, state):
    print("\nActive configuration:")
    print(f"- Mode: {mode}")
    print(f"- Ollama model: {LLM_MODEL}")
    print(f"- Retrieval k: {RETRIEVAL_K}")
    print(f"- Retrieval fetch_k: {RETRIEVAL_FETCH_K}")
    if mode == "Conversational RAG":
        print(f"- Source filter: {state.get('source_filter') or 'all documents'}")
    if mode == "Agentic RAG":
        status = "enabled" if state.get("web_search_enabled", True) else "disabled"
        print(f"- Web Search: {status}")


def _handle_source(argument, vectorstore, state):
    if not argument:
        print("Usage: /source <filename> or /source all")
        return
    if argument.lower() == "all":
        state["source_filter"] = None
        print("Source filter cleared. Searching all documents.")
        return

    try:
        documents = _documents(vectorstore)
    except Exception as error:
        print(f"Could not inspect documents: {type(error).__name__}: {error}")
        return

    matches = {document.lower(): document for document in documents}
    source = matches.get(argument.lower())
    if not source:
        print(f"Unknown document: {argument}")
        print("Use /docs to see available filenames.")
        return
    state["source_filter"] = source
    print(f"Source filter set to: {source}")


def _handle_web(argument, state):
    option = argument.lower()
    if option not in {"on", "off"}:
        print("Usage: /web on or /web off")
        return
    state["web_search_enabled"] = option == "on"
    print(f"Web Search {option}.")


def handle_command(query, vectorstore, clear_conversation, mode, state=None):
    """Handle a non-exit terminal command and return whether it was handled."""
    state = state if state is not None else {}
    command, _, argument = query.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command in {"/help", "help"}:
        print_help(mode)
        return True
    if command == "/status":
        run_startup_check()
        return True
    if command == "/config":
        _print_config(mode, state)
        return True
    if command in {"/docs", "docs"}:
        try:
            documents = _documents(vectorstore)
        except Exception as error:
            print(f"Could not list documents: {type(error).__name__}: {error}")
            return True
        print("\nIndexed documents:")
        for document in documents:
            print(f"- {document}")
        if not documents:
            print("(none)")
        return True
    if command in {"/clear", "clear"}:
        clear_conversation()
        print("Conversation cleared.")
        return True
    if command == "/source":
        if mode != "Conversational RAG":
            print("/source is available only in Conversational RAG mode.")
        else:
            _handle_source(argument, vectorstore, state)
        return True
    if command == "/web":
        if mode != "Agentic RAG":
            print("/web is available only in Agentic RAG mode.")
        else:
            _handle_web(argument, state)
        return True
    return False
