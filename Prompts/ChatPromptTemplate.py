from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

Chat_prompt = ChatPromptTemplate([
                                     ('system', "Let's you are a customar support agent"),
                                     ('human', '{Query}')
                                 ])

prompt = Chat_prompt.invoke({
                                'Query':"Where is my refound?"
                            })
print(prompt)
