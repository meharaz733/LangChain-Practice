from langchain_huggingface import (
        HuggingFaceEndpoint
    )
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import (
                create_react_agent,
                AgentExecutor
        )
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "openai/gpt-oss-120b",
    task = "text-generation",
    max_new_tokens=512
)

tools = [DuckDuckGoSearchRun()]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executer = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

response = executer.invoke({
                                     "input":"3 way to reach goa from delhi"
                                 })

print(response)


