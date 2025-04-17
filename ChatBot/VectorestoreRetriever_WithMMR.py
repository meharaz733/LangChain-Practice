from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(),
)

query = "What is langchain?"

retriever = vector_store.as_retriever(
                                      search_type='mmr',
                                      search_kwargs={'k':3, "lambda_mult": 0}
                                  )

result = retriever.invoke(query)

print(result)

#print(type(result))
#print (len(result))
#print(docs)
#print(result[1].page_content)
#print(type(docs[0].page_content))
