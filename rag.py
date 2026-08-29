from langchain_core.runnables import (RunnablePassthrough,RunnableParallel,RunnableLambda)

def format_docs(docs):#提取出document中的文本内容
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def get_sources(docs): #提取出document中的source内容
    return list({
        doc.metadata["source"]
        for doc in docs
    })

def create_rag_chain(retriever, prompt, llm):
    def retrieve(query):
        return retriever.invoke(query)
    
    def build_context(data):
        return format_docs(data["docs"])
    
    def build_sources(data):
        return get_sources(data["docs"])

    chain = (
        {
            "docs": retriever,#把传入检索后返回两个document元素的list
            "question": RunnablePassthrough()
        }
        | RunnableParallel(
            context=RunnableLambda(build_context), #传入docs生成context
            sources=RunnableLambda(build_sources), #传入docs过滤出其中的source
            question=lambda x: x["question"] 
        )
        | RunnableParallel(
            answer={
                "context": lambda x: x["context"],
                "question": lambda x: x["question"]
            }
            | prompt
            | llm,
            sources=lambda x: x["sources"]
        )
    )
    return chain