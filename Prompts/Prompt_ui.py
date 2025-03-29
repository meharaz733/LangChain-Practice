from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
model = ChatOpenAI()

paper_names = ['__SELECT__','A Mathematical Theory of Communication', 'The Chemical Basis of Morphogenesis', 'ImageNet Classification with Deep Convolutional Neural Networks', 'Deep Residual Learning for Image Recognition', 'Attention Is All You Need']
summarize_styles = ['Beginner-Friendly', "Layman’s", 'Code-Oriented', 'Technical Deep Dive']
summarize_lengths = ['Small - (1-2 paragraph)', 'Medium - (3-5 paragraph)', 'long - (detailed Explaination)']

st.header('Reasearch Tool')

paper_name = st.selectbox('Select the paper you want to summarize.', paper_names)
summarize_style = st.selectbox("Select Summerize style.", summarize_styles)
summarize_length = st.selectbox('Select the length', summarize_lengths)

#Prompt Tamplate
Prompt_Tamplate = load_prompt('Prompt_template.json')

#fill the Prompt_Tamplate
#prompt = Prompt_Tamplate.invoke({
#                                    "paper_name":paper_name,
#                                    "summarize_style": summarize_style,
#                                    "summarize_length": summarize_length
#                                })

if st.button('Summarize'):
    chain = Prompt_Tamplate | model
    result = chain.invoke({
                              "paper_name": paper_name,
                              "summarize_style": summarize_style,
                              "summarize_length":summarize_length
                          })
    st.write(result.content)
