#定义一个文本切分的函数-按照自然语言结构切分
def split_text(text, chunk_size=100):
    paragraphs = text.split("\n\n") #按照两个换行符切分
    chunks = []
#把文本先按段落切分，太长再按照句子切分，尽量把多个句子组合成不超过chunk_size的chunk
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            sentences = paragraph.split(". ") #把段落按句子切分
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + ". "
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence + ". "
            if current_chunk:
                chunks.append(current_chunk)
    return chunks
