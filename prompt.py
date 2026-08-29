from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

#只负责把问题写明白
def get_rewrite_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
Given the conversation history and the latest
user question, rewrite the latest question into
a standalone question that can be understood
without the conversation history.

Do not answer the question.
Only rewrite it when necessary.
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

#根据context回答
def get_answer_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """
Use the following retrieved context to answer
the user's question.

If the answer cannot be found in the context,
say "I don't know based on the provided documents."

Context:
{context}
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])