from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm_model = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm_model)

prompt = PromptTemplate(
    template='Make a standard note for following text: "{text}".',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'Make 5 questions for following text: "{text}".',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Marge the note and questions for a document. Answer all the questions if you can.\nNote: "{notes}.",\nQuestions: "{questions}."',
    input_variables = ['notes', 'questions']
)

parser = StrOutputParser()

#parallel chain create........
p_chain = RunnableParallel({
                               'notes': prompt | model | parser,
                               'questions': prompt2 | model | parser
                           })
s_chain = prompt3 | model | parser

chain = p_chain | s_chain

text = '''
A Machine Learning Engineer (ML Engineer) designs, builds, and deploys software that automates artificial intelligence and machine learning models, essentially bridging the gap between data scientists and the practical implementation of AI/ML systems. 
Here's a more detailed breakdown:

    What they do:
        Focus: ML engineers focus on the practical aspects of machine learning, ensuring models are robust, scalable, and can be deployed in real-world applications. 

Responsibilities: They work with data scientists to understand models, implement them in production, and maintain the systems. 
Skills: They need strong programming skills, knowledge of machine learning algorithms, and an understanding of software engineering principles. 
Tasks: This includes data preprocessing, model training, evaluation, deployment, and ongoing maintenance. 

How to become one:

    Education: A strong background in computer science, mathematics, or statistics is beneficial. 

Programming skills: Proficiency in languages like Python, R, or Java is crucial. 
Machine learning knowledge: Understanding various algorithms and models is important. 
Experience: Gaining hands-on experience through internships, projects, or entry-level positions is valuable. 
Continuous learning: The field of machine learning is constantly evolving, so continuous learning is essential. 

Salary:
Machine learning engineers are typically well-compensated, with salaries ranging from ₹3.0 Lakhs to ₹23.8 Lakhs per year in India. 
Difference between AI and ML Engineers:
AI engineers work on a broader set of tasks that encompass various forms of machine intelligence, while ML engineers focus more on ML algorithms and models that can self-tune to better learn and make predictions from large data sets.  
'''
result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()
