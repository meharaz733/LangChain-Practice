from langchain_community.vectorstores import Chroma, FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation',
)

#model = ChatHuggingFace(llm = llm_model,)

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=HuggingFaceEmbeddings(),
)

query = "How to improve energy levels and maintain balance?"

similarity_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

multiQuery_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {'k':5}),
    llm = llm_model,
)

s_result = similarity_retriever.invoke(query)
m_result = multiQuery_retriever.invoke(query)

print(s_result, "\n\n")
print(m_result)

#print(type(result))
#print (len(result))
#print(docs)
#print(result[1].page_content)
#print(type(docs[0].page_content))
