from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm_model = HuggingFaceEndpoint(repo_id = 'google/gemma-2-2b-it', task = 'text-generation')

model = ChatHuggingFace(llm=llm_model)

class Person(BaseModel):
    name: str = Field(description='The name of the person.')
    age: int = Field(description='The age of the person.')
    city: str = Field(description='The city of the person.')

parser = PydanticOutputParser(pydantic_object = Person)

prompt_Template = PromptTemplate(
    template = 'Generate a JSON object with the name, age, and city of a fictional Bangladeshi person. Only return the JSON object without explanations or additional text.\n{format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

chain = prompt_Template|model|parser

result = chain.invoke({})

print(result)
