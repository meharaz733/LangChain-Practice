from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

Chat_History = []

#Create Chat Prompt Template
Chat_prompt = ChatPromptTemplate([
                                     ('system', "Let's you are a customar support agent"),
                                     MessagesPlaceholder(variable_name='Chat_History'),
                                     ('human', '{Query}')
                                 ])
#load Chat History
with open('Chat_History.txt') as chatFile:
    Chat_History.extend(chatFile.readlines())

#print(Chat_History)

#Fill the Template
prompt = Chat_prompt.invoke({
                                'Chat_History': Chat_History,
                                'Query':"Where is my refound?"
                            })
print(prompt)
