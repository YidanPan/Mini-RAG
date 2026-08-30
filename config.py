# 综合汇总配置信息

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "qwen2.5:3b"

DATA_PATH = BASE_DIR / "data"

VECTORSTORE_PATH = BASE_DIR / "faiss_index"

CHUNK_SIZE = 200

CHUNK_OVERLAP = 40

RETRIEVAL_K = 2
RETRIEVAL_FETCH_K = 10
