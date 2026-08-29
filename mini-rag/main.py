import faiss
import numpy as np
import ollama
import json

from sentence_transformers import SentenceTransformer
from retrieval import retrieve

#main主要执行查询功能

with open(
    "index/documents.json",
    "r",
    encoding="utf-8"
) as f:
    documents = json.load(f)#现在存储的是json格式的documents，进行对应读取内容

model=SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("index/faiss.index")

query="What is Python used for?"
results=retrieve(query,documents,model,index,k=2)
if not results:
    print("No relevant information found.")
    exit()

context = "\n\n".join(
    result["text"]#results字典中读取每个text内容
    for result in results
)

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
answer = response["message"]["content"]

print("\nAnswer:")
print(answer)