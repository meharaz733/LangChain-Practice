from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature = 0.5, max_completion_tokens = 20)

result = model.invoke("Who am I?")

print(result.content)
