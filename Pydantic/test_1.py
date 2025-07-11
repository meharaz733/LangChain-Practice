from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Optional, Annotated

class Patient(BaseModel):
    name: Annotated[str, Field(max_length=50, title="Patient name", description="The name of the patient")]
    age: Annotated[int, Field(gt=0, strict=True)]
    email: EmailStr
    weight: float
    married: Optional[bool]
    allergies: List[str]
    contact_info: Dict[str, str]


def insert_patient_data(patient : Patient):
    print(patient.name)
    print(patient.age)
    print(patient.email)
    print(patient.allergies)
    print(patient.weight)
    print(patient.married)
    print(patient.contact_info['number'])

   
patient_info = {'age': 23, 'name': "Meharaz Hossain", 'weight': 5.6, 'married': True, 'allergies': ['dust'], 'contact_info': {'number': '01765053037','email':'meharaz733@gmail.com', 'tg':'rpyanthony10'}, 'email':'amv4@gmail.ccom'}

patient_1 = Patient(**patient_info)


insert_patient_data(patient_1)

