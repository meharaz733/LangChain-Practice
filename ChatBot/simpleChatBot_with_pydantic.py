from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import sys

load_dotenv()

#model = ChatOpenAI()

chat_history = [
    SystemMessage(content='Let, you are an AI assistent'),
    HumanMessage(content='Okay')
]

print(chat_history)
print(type(chat_history[0]))
print(type(chat_history))

#while True:
    # user_input = input('You: ')
    # chat_history.append(HumanMessage(content= user_input))
    # user_input = user_input.lower()
    
    # if(user_input=='exit'):
    #     sys.exit()

#    result = model.invoke(user_input)
#    print('AI: ', result.content)
#    chat_history.append(AIMessage(content=result.content))
    
