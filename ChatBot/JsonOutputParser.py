from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm_model = HuggingFaceEndpoint(repo_id = 'google/gemma-2-2b-it', task = 'text-generation')

model = ChatHuggingFace(llm=llm_model)

parser = JsonOutputParser()

temp = PromptTemplate(
    template = 'Give me the total number of Bangladesh, average old of people, total number of man, total number of woman.\n{format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction':parser.get_format_instructions()}
)

# prompt = temp.invoke(
#     {
        
#     }
# )

# result_ = model.invoke(prompt)

# result = parser.parse(result_.content)

#for line 20 to 28

Chain = temp|model|parser

result = Chain.invoke({})

print(result)

#print(prompt.text)

# Chain = temp1|model|parser|temp2|model|parser

# result = Chain.invoke({'topic':"United State"})
