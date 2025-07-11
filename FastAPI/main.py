from fastapi import FastAPI, Path, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import List, Annotated, Literal, Optional
import json

app = FastAPI()

class Patient(BaseModel):
    id: Annotated[str, Field(..., description='The patient ID', examples=['P001'])]
    name: Annotated[str, Field(..., description='The name of the patient', max_length=50, strict=True)]
    city: Annotated[str, Field(..., description='The city name where the patient is living')]
    age: Annotated[int, Field(..., gt=0, lt=150, description='The age of the patient')]
    gender: Annotated[Literal['male', 'female', 'third gender'], Field(..., description='Gender of the patient')]
    height: Annotated[float, Field(..., gt=0, description='The height of the patient in meters')]
    weight: Annotated[float, Field(..., gt=0, description='The weight of the patient in kg')]

    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round(self.weight/(self.height**2), 3)
        return bmi

    @computed_field
    @property
    def verdict(self) -> str:
        if self.bmi < 18.5:
            return 'Underweight'
        elif self.bmi < 25:
            return 'Normal'
        elif self.bmi < 30:
            return 'Normal'
        else:
            return 'Obese'

class PatientUpdate(BaseModel):
    name : Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None)]
    gender: Annotated[Optional[Literal['male', 'female', 'third_gender']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None)]
    weight: Annotated[Optional[float], Field(None)]

def load_data():
    with open('patients.json', 'r') as f:
        data = json.load(f)

        return data

def save_data(data):
    with open('patients.json', 'w') as f:
        json.dump(data, f)


@app.get('/')
def hello():
    return {'message': "patient managment system API."}

@app.get('/about')
def about():
    return {'message': "Fully functional API to manage patient data." }

@app.get('/view')
def view():
    data = load_data()

    return data

@app.get('/sort')
def sorted_data(sort_by : str = Query(..., description = 'Sort the patient date based on height, weight or BMI'), order : str = Query('asc', description = 'Ordering the data based on asc or desc')):
    
     valid_fields = ['height', 'weight', 'bmi']

     if sort_by not in valid_fields:
         raise HTTPException(status_code = 400, detail = 'Invalid parameter passed via sort_by')

     if order not in ['asc', 'desc']:
         raise HTTPException(status_code = 400, detail = 'Invalid parameter passed via order')

     data = load_data()

     order_status = True if order == 'desc' else False

     sorted_data = sorted(data.values(), key = lambda x: x.get(sort_by, 0), reverse= order_status)

     return sorted_data
 
@app.get('/patient/{patient_id}')
def patient_details(patient_id: str = Path(..., description = 'ID of the patient in the Database', example = 'P001')):
    data = load_data()

    if patient_id in data:
        return data[patient_id]
    raise HTTPException(status_code = 404, detail = 'Patient Not Found')


@app.post('/create')
def add_new_patient_data(patient: Patient):
    data = load_data()

    if patient.id in data:
        raise HTTPException(status_code=400, detail='The patient data already exist')

    data[patient.id] = patient.model_dump(exclude=['id'])

    save_data(data)

    return JSONResponse(status_code=201, content={'message':'Patient data added'})
 

@app.put('/edit/{patient_id}')
def edit_patient_data(patient_id: str, req_body: PatientUpdate):
    data = load_data()
    if patient_id not in data:
        raise HTTPException(status_code=404)

    existing_data = data[patient_id]

    update_patient_data = req_body.model_dump(exclude_unset=True)

    for key, value in update_patient_data.items():
      existing_data[key] = value

    existing_data['id'] = patient_id
    data_in_pydantic_obj = Patient(**existing_data)

    final_patient_data = data_in_pydantic_obj.model_dump(exclude=['id'])

    data[patient_id] = final_patient_data

    save_data(data)

    return JSONResponse(status_code=200, content={'message': 'Patient data updated'})

@app.delete('/delete/{patient_id}')
def delete_patient_data(patient_id: str):
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404)
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200, content={'message': 'Patient data deleted'})
