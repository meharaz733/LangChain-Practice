from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import requests

load_dotenv()

#tool creation....
#

@tool
def multifly(a: int, b: int) -> int:
    '''Given two numbers a nd b, then the function will return their product.'''
    return a*b

print(multifly.invoke({'a': 10, 'b': 20}))

print(multifly.name)
print(multifly.description)

print(multifly.args)


#tool Binding....

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation',
)

model = ChatHuggingFace(llm = llm_model)

model_with_tools = model.bind_tools([multifly])


#tool calling...

res1 = model_with_tools.invoke('Hi, how are you?') #llm will not suggest any tool for this query...

res2 = model_with_tools.invoke('can you multiply 45 and 9876532?') #llm will suggest the multiply tool...



#tool execution....

ans = multifly.invoke(res2.tool_calls[0])

print(ans)

