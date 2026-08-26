import numpy as np
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2") #加载一个训练好的模型
sentences=["Python is a programming language.","Python is used for software development.","I like eating apples."]
embedding=model.encode(sentences)#把文本转换为向量
print("embedding shape：",embedding.shape)

#提取出三个向量
python_vector=embedding[0]
software_vector=embedding[1]
apple_vector=embedding[2]

#相似度计算
similarity_python_software=np.dot(python_vector,software_vector)/(np.linalg.norm(python_vector)*np.linalg.norm(software_vector))

similarity_python_apple=np.dot(python_vector,apple_vector)/(np.linalg.norm(python_vector)*np.linalg.norm(apple_vector))

print("Python ↔ Software:", similarity_python_software)
print("Python ↔ Apple:",similarity_python_apple)