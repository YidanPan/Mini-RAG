from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

prompt = PromptTemplate(
    template="""
Answer the question briefly.

Question:
{question}
""",
    input_variables=["question"]
)

llm = ChatOllama(
    model="qwen2.5:3b"
)

chain = prompt | llm

response = chain.invoke({
    "question": "What is Python?"
})

print(response.content)