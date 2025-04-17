from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

docs = retriever.invoke("Dr. Muhammad Yunus and his intermediate government")

print(len(docs))
#print(docs)

print(docs[1].page_content)

print(type(docs[0].page_content))
