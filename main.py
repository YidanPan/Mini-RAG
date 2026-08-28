import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer
from retrieval import retrieve

#main主要执行查询功能

chunks = np.load("index/chunks.npy",allow_pickle=True).tolist() #直接导入存储在os中的

model=SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("index/faiss.index")

query="What is Python used for?"
results=retrieve(query,chunks,model,index,k=2)

context="\n\n".join(results)

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