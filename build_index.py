import faiss
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from loader import load_document
from chunk import split_text

#该文件主要实现：读取文档，进行切片转换为向量保存，建立索引并保存

all_chunks=[]

files=os.listdir("data") #是data路径下所有文件名的list
for file in files:
    if file.endswith(".txt"):
        path=os.path.join("data",file) #Python 会自动处理不同操作系统的路径，当然也可以直接用字符串拼接构造path
        text=load_document(path)
        chunks = split_text(text, chunk_size=100)
        all_chunks.extend(chunks) #把内容合并入list，最终得到一个list

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings) #建立索引

#保存到硬盘
os.makedirs("index",exist_ok=True) #在操作系统中建立一个文件目录来存储faiss索引
faiss.write_index(index,"index/faiss.index")

np.save(
    "index/chunks.npy",
    np.array(all_chunks, dtype=object)
) #保存chunks

print("Number of vectors:", index.ntotal)
print("FAISS index saved!")