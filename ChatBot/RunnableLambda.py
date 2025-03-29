from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm_model)

prompt = PromptTemplate(
    template = 'Write 3 line about the {topic}',
    input_variables = ['topic'],
    validate_template = True
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt, model, parser)

chain2 = RunnableParallel({
                             'text': RunnablePassthrough(),
                             'word_count': RunnableLambda(lambda x: len(x.split()))
                         })

chain3 = RunnableSequence(chain1, chain2)

result = chain3.invoke({
                          'topic':"human"
                      })

print(result)
