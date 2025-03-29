from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm_model)

prompt = PromptTemplate(
    template='Write interesting facts about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'Extract main 5 point of following text "{text}".',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Football'})

print(result)

chain.get_graph().print_ascii()
