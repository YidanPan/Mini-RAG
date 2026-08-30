from langgraph.checkpoint.memory import (InMemorySaver)#langchain中自带的memory功能
from embedding import get_embeddings
from vectorstore import load_vectorstore
from llm import get_llm
from agent import create_rag_agent
from agent_stream import stream_agent

def main():
    print("Loading Agentic RAG...")

    embeddings = get_embeddings()

    vectorstore = load_vectorstore(
        embeddings
    )

    llm = get_llm()

    checkpointer = InMemorySaver() #创建了一个内存状态保存器

    agent = create_rag_agent(
        llm=llm,
        vectorstore=vectorstore,
        checkpointer=checkpointer
    )

    config = {
        "configurable": {
            "thread_id":#当前对话的唯一编号
                "mini-rag-session"
        }
    }

    print("Agent ready.")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        stream_agent(agent=agent,query=query,config=config)

if __name__ == "__main__":
    main()