from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Annotated

load_dotenv()

model = ChatOpenAI()

#Schema
class Reviwe(BaseModel):
    Summary: str
    sentiment: str
    Key_theme: list[str] = Field(description= "Store key theme of the review.")
    name: Optional[str] = Field(description="return the name of reviewer.")

model2 = model.with_structured_output(Reviwe)

result = model2.invoke("""The XPhone Pro 12 offers a 6.8-inch OLED display, Snapdragon X1 chip,
                      108MP camera, and 5,000mAh battery with 65W fast charging.
                      It’s powerful and premium but slightly bulky.
                      Great for media and gaming!
                      """
                  )

print(result)
