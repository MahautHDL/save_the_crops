# TODO: Import your package, replace this by explicit imports of what you need
#from packagename.main import predict

import uvicorn
#import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import crops_package.model
#from crops_package.data import preprocessor
#from models.registry import load_model

app = FastAPI()
#app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': 'Hi, The API is running!'
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
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
@app.post('/upload')
def prediction_file(file:UploadFile ):
    '''pp_dogcat_image = preprocessing.image.load_img(dogcat_img, target_size=(150, 150))
    pp_dogcat_image_arr = preprocessing.image.img_to_array(pp_dogcat_image)
    input_arr = np.array([pp_dogcat_image_arr])
    prediction = np.argmax(model.predict(input_arr), axis=-1)
    print(prediction)'''
    '''X_pred = ----wouter
    X_processed = preprocessor(X_pred) --wouter
    y_pred = app.state.model.predict(X_processed)'''


    #return {'outcome': y_pred}

    name =file.filename
    return {'filename': name}

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
