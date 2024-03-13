# TODO: Import your package, replace this by explicit imports of what you need
#from packagename.main import predict
import os
import uvicorn
from io import BytesIO
import numpy as np
#import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from crops_package.data import prepocessor_img
from crops_package.registry import load_model

app = FastAPI()
app.state.model = load_model(plant='all')
app.state.cashew_model = load_model(plant='all')
app.state.cassava_model = load_model(plant='all')
app.state.maize_model = load_model(plant='all')
app.state.tomato_model = load_model(plant='all')


# app.state.cashew_model = load_model(plant='cashew')
# app.state.cassava_model = load_model(plant='cassava')
# app.state.maize_model = load_model(plant='maize')
# app.state.tomato_model = load_model(plant='tomato')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def root():
    return {
        'message': 'Hi, The API is running!'
    }


'''
@app.get("/predict")
def get_predict(input_one: float,
            input_two: float):
    # TODO: Do something with your input
    # i.e. feed it to your model.predict, and return the output
    # For a dummy version, just return the sum of the two inputs and the original inputs
    prediction = float(input_one) + float(input_two)
    return {
        'prediction': prediction,
        'inputs': {
            'input_one': input_one,
            'input_two': input_two
        }
    }
'''

@app.post('/uploadfile')
async def create_upload_file(plant: str = Form(...), file: UploadFile = File(...)):

    all_class_names = ['Cashew anthracnose','Cashew gumosis','Cashew healthy','Cashew leaf miner','Cashew red rust',
                        'Cassava bacterial blight','Cassava brown spot','Cassava green mite','Cassava healthy','Cassava mosaic',
                        'Maize fall armyworm','Maize grasshoper','Maize healthy','Maize leaf beetle','Maize leaf blight','Maize leaf spot','Maize streak virus',
                        'Tomato healthy','Tomato leaf blight','Tomato leaf curl','Tomato septoria leaf spot','Tomato verticulium wilt']
    cashew_class_names = ['Cashew anthracnose','Cashew gumosis','Cashew healthy','Cashew leaf miner','Cashew red rust']
    cassava_class_names = ['Cassava bacterial blight','Cassava brown spot','Cassava green mite','Cassava healthy','Cassava mosaic']
    maize_class_names = ['Maize fall armyworm','Maize grasshoper','Maize healthy','Maize leaf beetle','Maize leaf blight','Maize leaf spot','Maize streak virus']
    tomato_class_names = ['Tomato healthy','Tomato leaf blight','Tomato leaf curl','Tomato septoria leaf spot','Tomato verticulium wilt']


    class_names = {
        'all': all_class_names,
        'cashew': cashew_class_names,
        'cassava': cassava_class_names,
        'maize': maize_class_names,
        'tomato': tomato_class_names,
    }


    models_names = {
        'all': app.state.model,
        'cashew': app.state.cashew_model,
        'cassava': app.state.cassava_model,
        'maize': app.state.maize_model,
        'tomato': app.state.tomato_model,
    }

    # async to use the await
    input_image = await file.read()

    # convert the image to a tensor and resize it
    img = prepocessor_img(input_image)


    predictions = models_names[plant].predict(img)
    outcome = class_names[plant][np.argmax(predictions)].split()
    plant_disease = " ".join(outcome[1:])

    return {'disease' : plant_disease}


@app.post('/upload')
async def prediction_file(file:UploadFile=File(...)):

    class_names = ['Cashew anthracnose','Cashew gumosis','Cashew healthy','Cashew leaf miner','Cashew red rust',
                'Cassava bacterial blight','Cassava brown spot','Cassava green mite','Cassava healthy','Cassava mosaic',
                'Maize fall armyworm','Maize grasshoper','Maize healthy','Maize leaf beetle','Maize leaf blight','Maize leaf spot','Maize streak virus',
                'Tomato healthy','Tomato leaf blight','Tomato leaf curl','Tomato septoria leaf spot','Tomato verticulium wilt']

    # async to use the await
    input_image = await file.read()

    # convert the image to a tensor and resize it
    img = prepocessor_img(input_image)

    predictions = app.state.model.predict(img)
    outcome = class_names[np.argmax(predictions)].split()
    plant_name = outcome[0]
    plant_disease = " ".join(outcome[1:])

    return {'plant': plant_name, 'disease' : plant_disease}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
