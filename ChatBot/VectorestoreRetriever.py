from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader(file_path='Docs/Deep Learning Curriculum.pdf')

docs = loader.load()

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(),
    collection_name='my_collection'
)

query = "Tell me about loss function."

retriever = vector_store.as_retriever(search_kwargs={'k':2})

result = retriever.invoke(query)

print(result)

print(type(result))
print (len(result))
#print(docs)

print(result[1].page_content)

#print(type(docs[0].page_content))
