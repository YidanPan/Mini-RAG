import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunk import split_text
from retrieval import retrieve

text = """
Python is a programming language.
Python is widely used in artificial intelligence.
Python is also used in data science.
Python has a simple and readable syntax.
"""

chunks=split_text(text,chunk_size=100)
model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

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

print("Context:")
print(context)
