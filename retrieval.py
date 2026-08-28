import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from chunk import split_text

# 实现检索功能，传入问题和切片保存后的document

def retrieve(query,documents,model,index,k=2,threshold=1.0): #k决定最终到返回几个结果
    query_embedding=model.encode([query])
    query_embedding=np.array(query_embedding).astype("float32")
    distances,indices=index.search(query_embedding,k)

    results=[]
    for distance,i in zip(distances[0],indices[0]):
        if distance<=threshold:
            results.append(documents[i]) #document中索引和文本切片内容是对应的
    return results