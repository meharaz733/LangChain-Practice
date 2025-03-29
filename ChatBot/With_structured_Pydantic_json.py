from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

model = ChatOpenAI()

#Schema
# class Review(BaseModel):
#     Summary: str
#     sentiment: str
#     Key_theme: list[str] = Field(description= "Store key theme of the review.")
#     name: Optional[str] = Field(description="return the name of reviewer.")

Review = {
    "title":"Review",
    "description":"Structured information of the review",
    "type":"object",
    "properties":{
        "Summary":{
            "type":"string",
            "description":"Summary of the review."
        },
        "sentiment":{
            "type":"string",
            "enum":["pos", "neg"],
            "description":"Return pos for positive review and neg for negative review"
        },
        "Key_theme":{
            "type":"array",
            "items":{
                "type":"string"
            },
            "description":"Return all the key theme of the review."
        },
        "name":{
            "type":["string", "null"],
            "description":"store the reviewer name here if available"
        }
    },
    "required":["Summary", "sentiment", "Key_theme"]
}

model2 = model.with_structured_output(Review)

result = model2.invoke("""The XPhone Pro 12 offers a 6.8-inch OLED display, Snapdragon X1 chip,
                      108MP camera, and 5,000mAh battery with 65W fast charging.
                      It’s powerful and premium but slightly bulky.
                      Great for media and gaming!
                      """
                  )

print(result)
