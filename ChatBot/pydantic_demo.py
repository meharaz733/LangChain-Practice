# #Simple demo...
# from pydantic import BaseModel

# class Person(BaseModel):
#     name: str
#     age: int
#     roll: int
#     cgpa: float

# Meharaz_Hossain = {
#                     "name": "Meharaz Hossain",
#                     "age": 23.0,
#                     "roll": 21225103520,
#                     "cgpa": 3
# }

# stu_var = Person(**Meharaz_Hossain)

# print(stu_var)



#############################

#Defualt value set and optional

from pydantic import BaseModel, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'Meharaz'
    age: int
    cgpa: float = Field(gt= 0, lt=80)  #Value range should be 1-9 

student =  {
    'name': "Meharaz Hossain",
    'age': '10010120',    #Pydantic are able to detect int even we send the int value as str; :)
    'cgpa': 10
}

New_student = Student(**student)

stu_json = New_student.model_dump_json()

print(stu_json)
