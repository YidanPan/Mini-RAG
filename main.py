from embedding import get_embeddings
from vectorstore import load_vectorstore
from retriever import get_retriever
from prompt import get_prompt
from llm import get_llm
from rag import create_rag_chain

#主程序运行，实现完整RAG功能
def main():
    print("Loading vector store...")

    embeddings = get_embeddings()

    vectorstore = load_vectorstore(embeddings)

    source = input("Source filter (press Enter for all documents): ").strip()#根据用户输入来决定检索来源
    if source == "":
        source = None
    retriever = get_retriever(vectorstore,source=source)

    prompt = get_prompt()

    llm = get_llm()

    rag_chain = create_rag_chain(
        retriever,
        prompt,
        llm
    )

    while True:
        query = input("\nQuestion: ")
        if query.lower() == "exit":
            break

        results=rag_chain.invoke(query)

        print("\nAnswer:")
        print(results["answer"].content)

        print("\nSources:")
        for source in results["sources"]:
            print("-",source)

if __name__ == "__main__":
    main()