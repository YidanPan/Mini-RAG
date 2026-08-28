import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from loader import load_document
from chunk import split_text

import os

text = load_document("data/python.txt") #读取文档，进行切片转换为向量保存，建立索引并保存
chunks = split_text(text, chunk_size=100)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

os.makedirs("index",exist_ok=True) #在操作系统中建立一个文件目录来存储faiss索引
faiss.write_index(index,"index/faiss.index")

np.save(
    "index/chunks.npy",
    np.array(chunks, dtype=object)
) #保存chunks