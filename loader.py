from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from config import DATA_PATH

#加载所有txt文档
def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8"
        }
    )
    documents = loader.load()
    return documents