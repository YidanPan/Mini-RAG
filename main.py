from embedding import get_embeddings
from vectorstore import load_vectorstore
from retriever import get_retriever
from prompt import get_rewrite_prompt, get_answer_prompt
from llm import get_llm
from rag import conversational_rag_stream, message_text
from memory import ConversationMemory


def main():
    print("Loading vector store...")

    embeddings = get_embeddings()

    vectorstore = load_vectorstore(embeddings)

    source = input(
        "Source filter (press Enter for all documents): "
    ).strip()
    if source == "":
        source = None
    retriever = get_retriever(
        vectorstore,
        source=source
    )

    #初始化
    rewrite_prompt = get_rewrite_prompt()
    answer_prompt = get_answer_prompt()
    llm = get_llm()
    memory = ConversationMemory()

    while True:#一直循环实现多轮对话
        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue

        results = conversational_rag_stream(
            query=query,
            chat_history=memory.get_history(),
            retriever=retriever,
            rewrite_prompt=rewrite_prompt,
            answer_prompt=answer_prompt,
            llm=llm
        )
        print("\nStandalone Question:")
        print(results["standalone_question"])

        print("\nAnswer:")
        full_answer=""
        for chunk in results["stream"]:
            content=message_text(chunk.content)
            print(
                content,
                end="",
                flush=True
            )#实现streaming输出
            full_answer+=content
        print()

        print("\nSources:")
        for source in results["sources"]:
            print("-", source)

        memory.add_user_message(query)
        memory.add_ai_message(full_answer)#把拼接好的完整回答加入记忆库中

if __name__ == "__main__":
    main()
