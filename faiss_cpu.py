import faiss
import numpy as np

vectors=np.array([[1,0],[0,1],[1,1]],dtype="float32")
dimension=vectors.shape[1]
index=faiss.IndexFlatL2(dimension)

index.add(vectors)

print(index.ntotal)#有几个向量

query=np.array([[0.9,0.1]],dtype="float32")
distances,indices=index.search(query,k=2)#找最相似的两个向量 indices是这些向量在原始数组中的坐标

print("Distance:")
print(distances)

print("Indices:")
print(indices)

#faiss是一个存储和搜索向量的数据结构