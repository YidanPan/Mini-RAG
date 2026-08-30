"""Streamlit UI for Conversational RAG and Agentic RAG."""

import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from agent import create_rag_agent
from agent_stream import iter_agent_events
from embedding import get_embeddings
from llm import get_llm
from memory import ConversationMemory
from prompt import get_answer_prompt, get_rewrite_prompt
from retriever import get_retriever
from startup_check import collect_startup_status
from vectorstore import load_vectorstore

CONVERSATION_DIR = Path(__file__).resolve().parent / "conversation rag"
if str(CONVERSATION_DIR) not in sys.path:
    sys.path.insert(0, str(CONVERSATION_DIR))
from rag import conversational_rag_stream, message_text


st.set_page_config(page_title="Mini RAG", page_icon="📚", layout="wide")


TEXT = {
    "中文": {
        "language": "界面语言",
        "title": "Mini RAG",
        "caption": "本地知识库检索与可选联网搜索的智能 Agent。",
        "session": "会话设置",
        "mode": "运行模式",
        "conversational": "对话式 RAG",
        "agent": "Agentic RAG",
        "clear": "清空当前对话",
        "startup": "启动状态",
        "startup_error": "必要的本地服务不可用，请先根据启动状态修复问题。",
        "all_documents": "全部文档",
        "document_filter": "本地文档筛选",
        "web_search": "允许联网搜索",
        "web_search_hint": "联网搜索是可选功能，只有 Agent 主动调用时才会执行。",
        "ask_documents": "请输入关于知识库的问题…",
        "ask_agent": "请输入给 Agent 的任务…",
        "sources": "来源",
        "local_source": "本地：{source}",
        "web_source": "网页：{url}",
        "agent_working": "Agent 正在处理…",
        "tool_call": "工具：`{name}` — `{args}`",
        "tool_result": "工具结果：`{name}`",
        "agent_finished": "Agent 已完成",
        "agent_empty": "Agent 没有返回回答。",
        "error": "错误：{name}：{detail}",
    },
    "English": {
        "language": "Language",
        "title": "Mini RAG",
        "caption": "Local knowledge-base retrieval with an optional web-search Agent.",
        "session": "Session",
        "mode": "Mode",
        "conversational": "Conversational RAG",
        "agent": "Agentic RAG",
        "clear": "Clear conversation",
        "startup": "Startup status",
        "startup_error": "Required local services are unavailable. Fix the startup status first.",
        "all_documents": "All documents",
        "document_filter": "Local document filter",
        "web_search": "Allow Web Search",
        "web_search_hint": "Web Search is optional and is only used when the Agent calls it.",
        "ask_documents": "Ask about your documents…",
        "ask_agent": "Ask the Agent…",
        "sources": "Sources",
        "local_source": "Local: {source}",
        "web_source": "Web: {url}",
        "agent_working": "Agent is working…",
        "tool_call": "Tool: `{name}` — `{args}`",
        "tool_result": "Tool result: `{name}`",
        "agent_finished": "Agent finished",
        "agent_empty": "The Agent did not return an answer.",
        "error": "Error: {name}: {detail}",
    },
}


def tr(key, **kwargs):
    """Return the current interface language string."""
    language = st.session_state.get("language", "中文")
    return TEXT[language][key].format(**kwargs)


@st.cache_resource(show_spinner="Loading local models and vector store...")
def load_resources():
    embeddings = get_embeddings()
    return load_vectorstore(embeddings), get_llm()


def indexed_documents(vectorstore):
    return sorted(
        {
            document.metadata.get("source")
            for document in vectorstore.docstore._dict.values()
            if document.metadata.get("source")
        }
    )


def initialize_session():
    defaults = {
        "messages": [],
        "conversation_memory": ConversationMemory(),
        "source_filter": None,
        "web_search_enabled": True,
        "agent_thread_id": f"mini-rag-ui-{uuid4().hex}",
        "agent": None,
        "language": "中文",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_conversation():
    st.session_state.messages = []
    st.session_state.conversation_memory = ConversationMemory()
    st.session_state.agent_thread_id = f"mini-rag-ui-{uuid4().hex}"


def append_sources(content, sources):
    if not isinstance(content, str):
        return
    lines = content.splitlines()
    for line in lines:
        if line.startswith("[Source: "):
            source = line.removeprefix("[Source: ").split("]", 1)[0]
            sources["local"].add(source)
        elif line.startswith("URL: "):
            sources["web"].add(line.removeprefix("URL: "))


def render_sources(sources):
    if not sources["local"] and not sources["web"]:
        return
    with st.expander(tr("sources")):
        for source in sorted(sources["local"]):
            st.write(tr("local_source", source=source))
        for url in sorted(sources["web"]):
            st.markdown(tr("web_source", url=url))


def render_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["sources"])


def run_conversational_query(query, vectorstore, llm):
    retriever = get_retriever(vectorstore, source=st.session_state.source_filter)
    result = conversational_rag_stream(
        query=query,
        chat_history=st.session_state.conversation_memory.get_history(),
        retriever=retriever,
        rewrite_prompt=get_rewrite_prompt(),
        answer_prompt=get_answer_prompt(),
        llm=llm,
    )
    answer = ""
    placeholder = st.empty()
    for chunk in result["stream"]:
        answer += message_text(chunk.content)
        placeholder.markdown(answer + "▌")
    placeholder.markdown(answer)
    st.session_state.conversation_memory.add_user_message(query)
    st.session_state.conversation_memory.add_ai_message(answer)
    return answer, {"local": set(result["sources"]), "web": set()}


def get_agent(vectorstore, llm):
    if st.session_state.agent is None:
        runtime_options = {"web_search_enabled": st.session_state.web_search_enabled}
        st.session_state.agent_runtime_options = runtime_options
        st.session_state.agent = create_rag_agent(
            llm=llm,
            vectorstore=vectorstore,
            checkpointer=InMemorySaver(),
            runtime_options=runtime_options,
        )
    st.session_state.agent_runtime_options["web_search_enabled"] = (
        st.session_state.web_search_enabled
    )
    return st.session_state.agent


def run_agent_query(query, vectorstore, llm):
    agent = get_agent(vectorstore, llm)
    config = {"configurable": {"thread_id": st.session_state.agent_thread_id}}
    answer = ""
    sources = {"local": set(), "web": set()}
    answer_placeholder = st.empty()
    with st.status(tr("agent_working"), expanded=True) as status:
        for event in iter_agent_events(agent, query, config):
            if event["type"] == "tool_call":
                status.write(tr("tool_call", name=event["name"], args=event["args"]))
            elif event["type"] == "tool_result":
                if event["error"]:
                    status.error(event["content"])
                else:
                    status.write(tr("tool_result", name=event["name"]))
                    append_sources(event["content"], sources)
            elif event["type"] == "progress":
                status.write(event["message"])
            elif event["type"] == "answer_token":
                answer += event["text"]
                answer_placeholder.markdown(answer + "▌")
        status.update(label=tr("agent_finished"), state="complete", expanded=False)
    answer_placeholder.markdown(answer or tr("agent_empty"))
    return answer, sources


def main():
    initialize_session()
    records = collect_startup_status()
    required_ready = all(ok for _label, ok, _detail, required in records if required)
    with st.sidebar:
        st.segmented_control(
            "Language / 语言",
            ["中文", "English"],
            key="language",
            default="中文",
            required=True,
            width="stretch",
        )
        st.header(tr("session"))
        mode = st.radio(
            tr("mode"),
            ["conversational", "agent"],
            format_func=lambda option: tr(option),
            key="mode",
        )
        if st.button(tr("clear")):
            clear_conversation()
            st.rerun()

        st.header(tr("startup"))
        for label, ok, detail, required in records:
            icon = "✅" if ok else ("⚠️" if not required else "❌")
            st.write(f"{icon} {label}: {detail}")

    st.title(tr("title"))
    st.caption(tr("caption"))

    if not required_ready:
        st.error(tr("startup_error"))
        st.stop()

    vectorstore, llm = load_resources()
    documents = indexed_documents(vectorstore)
    with st.sidebar:
        if mode == "conversational":
            all_documents = tr("all_documents")
            options = [all_documents, *documents]
            selected_index = (
                options.index(st.session_state.source_filter)
                if st.session_state.source_filter in options
                else 0
            )
            selected = st.selectbox(
                tr("document_filter"),
                options,
                index=selected_index,
            )
            st.session_state.source_filter = None if selected == all_documents else selected
        else:
            st.toggle(tr("web_search"), key="web_search_enabled")
            st.caption(tr("web_search_hint"))

    render_history()
    placeholder = tr("ask_documents") if mode == "conversational" else tr("ask_agent")
    if query := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            try:
                if mode == "conversational":
                    answer, sources = run_conversational_query(query, vectorstore, llm)
                else:
                    answer, sources = run_agent_query(query, vectorstore, llm)
                render_sources(sources)
            except Exception as error:
                answer = tr("error", name=type(error).__name__, detail=error)
                sources = {"local": set(), "web": set()}
                st.error(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
