import faiss
import numpy as np
import os
import json

from sentence_transformers import SentenceTransformer
from loader import load_document
from chunk import split_text

#该文件主要实现：读取文档，进行切片转换为向量保存，建立索引并保存

all_chunks=[]#保存所有txt文档的切片（原文本）
documents=[]#保存原文本切片对应的索引和文本来源（Metadata）

files=os.listdir("data") #是data路径下所有文件名的list
for file in files:
    if file.endswith(".txt"):
        path=os.path.join("data",file) #Python 会自动处理不同操作系统的路径，当然也可以直接用字符串拼接构造path

        text=load_document(path)

        chunks = split_text(text, chunk_size=100)

        all_chunks.extend(chunks)
        for chunk_id,chunk in enumerate(chunks):
            documents.append({
                "text":chunk,
                "source":file,
                "chunk_id":chunk_id
            })

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings) #建立索引

#保存到硬盘
os.makedirs("index",exist_ok=True) #在操作系统中建立一个文件目录来存储faiss索引
faiss.write_index(index,"index/faiss.index")

#json格式比较适合保存document中内容格式
with open(
    "index/documents.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )