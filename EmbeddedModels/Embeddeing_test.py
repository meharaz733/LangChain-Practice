import asyncio
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

#async def main():
Embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions = 32)

documents = [
                "Hello, my name is Meharaz Hossain",
                "I am a student of Bsc. in CSE",
                "My institute name is BUBT"
        ]

#result = Embedding.embed_query('What is the name of dash')
#result_2 = Embedding.embed_documents(documents)
print(type(documents))
#print(str(result))
#print(str(result_2))

#asyncio.run(main())

