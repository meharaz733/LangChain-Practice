from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

#embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

loader = PyPDFLoader(file_path='../ChatBot/Docs/Deep Learning Curriculum.pdf')

docs = loader.load()

vector_store = Chroma(
                      embedding_function=HuggingFaceEmbeddings(),
                      persist_directory='chromadb2',
                      collection_name='sample'
                      )


kei_id = vector_store.add_documents(docs)

#print(vector_store)

#print(kei_id)
#

get_value = vector_store.get(include=['embeddings', 'documents'])

query = "Tell me about loss function."

#print(len(get_value['embeddings'][0]))

result = vector_store.similarity_search(query=query, k=1)

print(result)
