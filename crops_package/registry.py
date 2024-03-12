import glob
import os
import time
import pickle

# from colorama import Fore, Style
from tensorflow import keras

# from taxifare.params import *
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import Model
from tensorflow import keras


def load_model(stage="Production", plant='all') -> keras.Model:
    """
    Loads a saved model from MLflow, trying multiple tracking URIs in sequence.
    Returns the loaded model or None if no model is found after trying all tracking URIs.
    """

    model = None
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get all versions of the model filtered by name
    model_name = os.environ.get(f'MLFLOW_MODEL_{plant.upper()}')

    model_versions = client.get_latest_versions(name=model_name, stages=[stage])
    model_uri = model_versions[0].source
    model = mlflow.tensorflow.load_model(model_uri=model_uri)


    return model


#mlflow.set_tracking_uri("https://mlflow.lewagon.ai")


#model = mlflow.tensorflow.load_model(model_uri=model_uri)
