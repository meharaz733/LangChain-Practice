from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(repo_id = 'Qwen/QwQ-32B', task = 'text-generation')

model = ChatHuggingFace(llm=llm_model)

temp1 = PromptTemplate(template = "write detailed about {topic}", input_variables=['topic'])
temp2 = PromptTemplate(template='Write 5 line summary about this text "{text}"', input_variables=['text'])

#input1 = temp1.invoke({"topic":"United State"})

#result = model.invoke(input1)

#input2 = temp2.invoke({"text":result.content})

#result2 = model.invoke(input2)

#print(result2.content)

#With Output Parsers:

parser = StrOutputParser()

Chain = temp1|model|parser|temp2|model|parser

result = Chain.invoke({'topic':"United State"})
