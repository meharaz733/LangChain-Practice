from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm_model)

prompt1 = PromptTemplate(
    template = 'Write 1 line about the {topic}',
    input_variables = ['topic'],
    validate_template = True
)

prompt2 = PromptTemplate(
    template = 'Write 3 line about the {topic}',
    input_variables = ['topic'],
    validate_template = True
)

parser = StrOutputParser()

chain = RunnableParallel({
                             'first': RunnableSequence(prompt1, model, parser),
                             'second': RunnableSequence(prompt2, model, parser)
                         })

result = chain.invoke({
                          'topic':"human"
                      })

print(result)
