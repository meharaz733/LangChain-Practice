from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatOpenAI()

#Schema
class Reviwe(TypedDict):
    Summary: str
    sentiment: str

model2 = model.with_structured_output(Reviwe)

result = model2.invoke("""The XPhone Pro 12 offers a 6.8-inch OLED display, Snapdragon X1 chip,
                      108MP camera, and 5,000mAh battery with 65W fast charging.
                      It’s powerful and premium but slightly bulky.
                      Great for media and gaming!
                      """
                  )

print(result)
