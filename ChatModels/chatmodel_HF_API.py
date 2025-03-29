from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llmmodel = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text_generation"
)

model = ChatHuggingFace(llm = llmmodel)
result = model.invoke("Do you know about BUBT?")

print(result.content)
