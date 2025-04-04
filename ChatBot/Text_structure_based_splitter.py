from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = '''
LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.

History:
LangChain was launched in October 2022 as an open source project by Harrison Chase, while working at machine learning startup Robust Intelligence. The project quickly garnered popularity, with improvements from hundreds of contributors on GitHub, trending discussions on Twitter, lively activity on the project's Discord server, many YouTube tutorials, and meetups in San Francisco and London. In April 2023, LangChain had incorporated and the new startup raised over $20 million in funding at a valuation of at least $200 million from venture firm Sequoia Capital, a week after announcing a $10 million seed investment from Benchmark.

In the third quarter of 2023, the LangChain Expression Language (LCEL) was introduced, which provides a declarative way to define chains of actions.

In October 2023 LangChain introduced LangServe, a deployment tool to host LCEL code as a production-ready API.
'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=0
)

loader = PyPDFLoader('Docs/Deep Learning Curriculum.pdf')
docs = loader.load()

result1 = splitter.split_text(text)


result2 = splitter.split_documents(docs)

for i in range(len(result1)):
    print(result1[i],  '\n\n')
    
print(len(result1))

print('\n\n\n', result2[0], '\n')
print(len(result2))
print(result2[2].page_content)
