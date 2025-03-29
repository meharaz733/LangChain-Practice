from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm_model)

prompt1 = PromptTemplate(
    template = 'Explain interesting fact about {topic}',
    input_variables = ['words', 'topic'],
    validate_template = True
)

prompt2 = PromptTemplate(
    template = 'Extract main 5 point of the following text - {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser)

result = chain.invoke({
                          'words':200,
                          'topic':"human"
                      })

print(result)
