from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

em_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

doc_ =  ["Elon Musk – The billionaire entrepreneur behind Tesla, SpaceX, and Neuralink, pushing the boundaries of technology and space travel.",

"Taylor Swift – A global pop icon and singer-songwriter known for her record-breaking albums and deep connection with fans.",

"Cristiano Ronald – One of the greatest footballers of all time, winning multiple Ballon d'Or awards and breaking goal-scoring records.",

"Albert Einstein – The legendary physicist who developed the theory of relativity, changing the way we understand space and time.",

"Oprah Winfrey – A media mogul, talk show host, and philanthropist who inspired millions with her storytelling and generosity.",

"Narendra Modi – The Prime Minister of India, known for his leadership, economic reforms, and strong global presence."
]

query = input("Ask your Question: ")


em_doc = em_model.embed_documents(doc_)
em_query = em_model.embed_query(query)

final_result =  cosine_similarity([em_query], em_doc)[0]


index, score = sorted([(i, float(scores)) for i, scores in enumerate(final_result)], key = lambda x:x[1])[-1]

#print(sorted([(i, float(scores)) for i, scores in enumerate(final_result)], key = lambda x:x[1]))

#print(index, score)

print(doc_[index])
print("Scores: ", score)

