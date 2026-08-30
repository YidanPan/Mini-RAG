"""Startup checks shared by the conversational and Agentic RAG entry points."""

import json
import os
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

from config import BASE_DIR, DATA_PATH, LLM_MODEL, VECTORSTORE_PATH


def _ollama_models():
    with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
        payload = json.load(response)
    return {model.get("name") for model in payload.get("models", [])}


def collect_startup_status():
    """Return prerequisite records as (label, ok, detail, required)."""
    load_dotenv(BASE_DIR / ".env")
    document_count = len(list(DATA_PATH.rglob("*"))) if DATA_PATH.is_dir() else 0
    index_files = [VECTORSTORE_PATH / "index.faiss", VECTORSTORE_PATH / "index.pkl"]
    index_ready = all(path.is_file() for path in index_files)
    records = [
        ("Data directory", DATA_PATH.is_dir(), f"{document_count} entries", True),
        ("Vector store", index_ready, str(VECTORSTORE_PATH), True),
    ]
    try:
        models = _ollama_models()
        records.append(("Ollama service", True, "reachable", True))
        records.append(("Configured model", LLM_MODEL in models, LLM_MODEL, True))
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as error:
        records.append(("Ollama service", False, str(error), True))
        records.append(("Configured model", False, LLM_MODEL, True))
    tavily_ready = bool(os.getenv("TAVILY_API_KEY"))
    records.append(
        (
            "Web search",
            tavily_ready,
            "TAVILY_API_KEY configured" if tavily_ready else "optional: add TAVILY_API_KEY to .env",
            False,
        )
    )
    return records


def run_startup_check():
    """Print prerequisites and return whether required local services are ready."""
    print("\nStartup check")
    records = collect_startup_status()
    for label, ok, detail, _required in records:
        mark = "OK" if ok else "MISSING"
        print(f"[{mark}] {label}: {detail}")
    ready = all(ok for _label, ok, _detail, required in records if required)
    if not ready:
        print("Fix the missing required items above before starting the chat.")
    return ready
