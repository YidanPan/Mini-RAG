from langchain_community.document_loaders import (DirectoryLoader,TextLoader,UnstructuredMarkdownLoader)
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def normalize_metadata(documents):
    for doc in documents:
        source = doc.metadata.get("source")
        if source:
            doc.metadata["source"] = Path(source).name
            suffix = Path(source).suffix.lower()
            if suffix == ".pdf":
                doc.metadata["type"] = "pdf"
            elif suffix == ".txt":
                doc.metadata["type"] = "txt"
            elif suffix == ".md":
                doc.metadata["type"] = "markdown"
    return documents

#把各种文件转换为Langchain document。虽然输入格式不同，但进入 RAG 后统一成相同的数据结构
def load_documents(path):
    documents = []

    txt_loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    md_loader = DirectoryLoader(
        path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader
    )

    pdf_loader=DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents.extend(txt_loader.load())
    documents.extend(md_loader.load())
    documents.extend(pdf_loader.load())

    return documents
