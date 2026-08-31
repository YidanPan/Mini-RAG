# Agentic Mini RAG

## 中文简介

Mini RAG 是一个可在本地运行的 RAG（检索增强生成）学习项目。它将本地文档构建为 FAISS 向量知识库，再由 Ollama 本地模型基于检索结果回答问题；同时提供一个能够自主选择工具的 Agent 模式。

项目包含两种使用方式：终端 CLI 和 Streamlit 网页界面。网页侧边栏可以在中文与 English 间切换界面语言，且不会清空当前对话。

主要能力：

- 对话式 RAG：支持追问改写、文档检索、答案流式输出与会话记忆。
- Agentic RAG：可调用本地知识库搜索、指定文档搜索、文档列表、计算器，以及可选的 Tavily 联网搜索。
- 运行保障：包含启动自检、结构化工具错误信息、`.env.example` 与终端快捷指令。

适合用于理解 RAG、工具调用、Agent 工作流和本地大模型应用的基础结构。

## 中文使用方法

### 1. 准备环境

请先安装 Python 3.12+ 和 [Ollama](https://ollama.com/)，然后在项目目录执行：

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
ollama pull qwen2.5:3b
Copy-Item .env.example .env
```

如果需要 Agent 联网搜索，请在 `.env` 中填写 Tavily Key：

```text
TAVILY_API_KEY=你的_Tavily_API_Key
```

`.env` 只用于本机配置，不要上传到 GitHub。

### 2. 构建知识库

把 `.txt`、`.pdf` 等资料放入 `data/` 后，构建或更新 FAISS 向量库：

```powershell
& ".\.venv\Scripts\python.exe" .\build_index.py
```

每次修改 `data/` 中的文档后，都应重新执行这条命令。

### 3. 启动网页界面

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py
```

浏览器打开 `http://localhost:8501`。侧边栏可完成以下操作：

- `中文 / English`：切换界面语言。
- `对话式 RAG`：基于本地知识库连续问答，可选择只检索某一份文档。
- `Agentic RAG`：由 Agent 自行选择文档搜索、列出文档、计算器和联网搜索等工具。
- `允许联网搜索`：仅在 Agentic RAG 模式出现；关闭后 Agent 不会调用 Tavily。
- `清空当前对话`：清除当前浏览器会话中的聊天记录。

### 4. 启动终端模式（可选）

```powershell
# 对话式 RAG
& ".\.venv\Scripts\python.exe" ".\conversation rag\main.py"

# Agentic RAG
& ".\.venv\Scripts\python.exe" .\agent_main.py
```

常用终端指令：`/help`、`/status`、`/config`、`/docs`、`/clear`、`/exit`。
对话式 RAG 额外支持 `/source <文件名>` 和 `/source all`；Agentic RAG 支持 `/web on` 与 `/web off`。

---

## English overview

A local RAG learning project with two terminal modes:

- **Conversational RAG**: rewrites follow-up questions, retrieves from the local FAISS knowledge base, and streams grounded answers.
- **Agentic RAG**: lets an Ollama tool-calling agent choose local search, document-specific search, document listing, arithmetic, and optional Tavily Web Search.

## Prerequisites

- Python 3.12+
- Ollama running locally with `qwen2.5:3b`
- Optional: a Tavily API key for Web Search

## Setup

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add `TAVILY_API_KEY` to `.env` only if you want Web Search. Do not commit `.env`.

Build or rebuild the local vector store after changing files in `data/`:

```powershell
& ".\.venv\Scripts\python.exe" .\build_index.py
```

## Run

Start Conversational RAG:

```powershell
& ".\.venv\Scripts\python.exe" ".\conversation rag\main.py"
```

Start Agentic RAG:

```powershell
& ".\.venv\Scripts\python.exe" .\agent_main.py
```

Both modes run a startup check for the FAISS index, Ollama service, configured model, and optional Web Search key.

## Web UI

Launch the Streamlit interface:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run .\app.py
```

The UI provides a mode selector, streamed answers, tool-call progress, local and web sources, document filtering, Web Search control, startup status, and a clear-conversation button.

## Terminal commands

- `/help` — show commands
- `/status` — re-run the local startup check
- `/config` — show active non-secret configuration
- `/docs` — list local indexed documents
- `/clear` — clear the current conversation
- `/exit` — exit

Conversational RAG also supports `/source <filename>` and `/source all` to control its local-document filter. Agentic RAG supports `/web on` and `/web off` to control whether its Web Search tool may access Tavily.

`exit`, `clear`, `docs`, and `help` also work without `/`.

## Example prompts

- `According to my documents, what are the stages of RAG?`
- `List my documents.`
- `Search rag.txt for the retrieval stage.`
- `Calculate 12 multiplied by 3.`
- `Search the web for the latest Python release.`

## Project structure

- `data/` — source documents
- `build_index.py` — builds the FAISS index
- `conversation rag/` — Conversational RAG CLI and chain
- `agent_main.py` / `agent.py` — Agentic RAG CLI and Agent setup
- `tools.py` — local retrieval, calculator, and Web Search tools
- `startup_check.py` — shared startup prerequisites check
