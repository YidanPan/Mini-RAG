#定义一个文本切分的函数
def split_text(text,chunk_size=100,chunk_overlap=20):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-chunk_overlap
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