from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm_model)

parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Extract the main topic of the following text - "{docs}"',
    input_variables = ['docs'],
    validate_template = True
)

path = 'Docs/books'

loader = DirectoryLoader(
    path = path,
    glob = '*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'docs': docs[43].page_content})

print(docs[43])
#print(len(docs))

print(result)
