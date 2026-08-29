from langchain_core.prompts import PromptTemplate

#创建prompt模版
def get_prompt():

    return PromptTemplate(
        template="""
Use the following context to answer the question.

If the answer cannot be found in the context,
say "I don't know based on the provided documents."

Context:

{context}

Question:

{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )