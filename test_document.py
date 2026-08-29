from langchain_core.documents import Document

doc = Document(
    page_content="Python is a programming language.",
    metadata={
        "source": "python.txt"
    }
)

print(doc)
print(doc.page_content)
print(doc.metadata)