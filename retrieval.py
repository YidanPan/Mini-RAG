import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunk import split_text

def retrieve(query,chunks,model,index,k=2):
    query_embedding=model.encode([query])
    query_embedding=np.array(query_embedding).astype("float32")
    distances,indeces=index.search(query_embedding,k)
    results=[]
    for i in indeces[0]:
        results.append(chunks[i])
    return results