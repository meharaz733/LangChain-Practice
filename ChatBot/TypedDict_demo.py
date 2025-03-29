from typing import TypedDict

class person(TypedDict):
    Name: str
    Age: int

new_person: person = {
                        'Name': "Meharaz Hossain",
                        'Age': 19
                    }

print(new_person)

