#模块化实现输入路径能读取路径下的文档内容
def load_document(path):
    with open(path,"r",encoding="utf-8") as f:
        text=f.read()
    return text