from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    "data/python.txt",
    encoding="utf-8"
)

documents = loader.load() #document是一个list，有page_content/matedata

print("Number of documents:", len(documents))

print("\nContent:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)