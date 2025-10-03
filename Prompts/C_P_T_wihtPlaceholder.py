from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

Chat_History = []

# #Create Chat Prompt Template
Chat_prompt = ChatPromptTemplate([
                                     ('system', "Let's you are a customar support agent"),
#                                      # MessagesPlaceholder(variable_name='Chat_History'),
                                     ('human', '{Query}')
                                 ])
# #load Chat History
# # with open('Chat_History.txt') as chatFile:
# #     Chat_History.extend(chatFile.readlines())

print(Chat_History)

 #Fill the Template
prompt = Chat_prompt.invoke({
                                'Query':"Where is my refound?"
                            })
print(prompt)
#
exit()

list_of_tuple1 = [("system", "lets a you're bolod")]
list_of_tuple2 = [("human", "hi"), ("ai", "hello, how I can assist you today?")]

history = list_of_tuple1 + list_of_tuple2
Chat_prompt = ChatPromptTemplate(history+[("human", "{query}\nContent:{content:}")])
#load Chat History
# with open('Chat_History.txt') as chatFile:
#     Chat_History.extend(chatFile.readlines())

#print(Chat_History)

#Fill the Template
prompt = Chat_prompt.invoke({"query": "No, I don't need any assistment", "content": "Tahole msg dicho kno?"})
print(prompt)
