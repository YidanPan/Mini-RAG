#定义一个文本切分的函数
def split_text(text,chunk_size=100):
    chunks=[]
    for i in range(0,len(text),chunk_size):
        chunk=text[i:i+chunk_size] #每一个chunk的文本切片
        chunks.append(chunk)
    return chunks

text = """
Python is a programming language.
Python is widely used in artificial intelligence.
Python is also used in data science.
Python has a simple and readable syntax.
"""

chunks = split_text(text, chunk_size=50)

for i, chunk in enumerate(chunks):
    print(f"\n Chunk {i}:")
    print(chunk)