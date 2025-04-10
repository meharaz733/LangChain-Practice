from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pc = Pinecone()#api_key='pcsk_5cuBeR_GRrQ6KjQ19pHUM6FjV855t5VrXUrurcaMoWi6GXY82EiYLu6VfLp7C9PbZgtLAr')
# pc.create_index(
#                         name='sample',
#                         spec={
#                             "serverless": {
#                             "cloud": "aws",         # or "gcp"
#                             "region": "us-east-1"   # or your preferred region
#                             }
#                         },
#                         dimension= 384
#                     )
index = pc.Index(name='sample')

embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

loader = PyPDFLoader(file_path='../ChatBot/Docs/Deep Learning Curriculum.pdf')

docs = loader.load()

vector_store = PineconeVectorStore(
                      embedding=embedding_model,
                      index=index
                      )


kei_id = vector_store.add_documents(docs)

#print(vector_store)

#print(kei_id)


query = "Tell me about loss function."

#print(len(get_value['embeddings'][0]))

result = vector_store.similarity_search(query=query, k=1)

print(result)
