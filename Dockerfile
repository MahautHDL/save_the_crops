FROM python:3.10.6-buster

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY crops_package crops_package
COPY api api
COPY setup.py setup.py
RUN pip install .

# COPY models models
# ENV MODEL_PATH models/model_CNN8_62_2.keras

# CMD uvicorn api.fast:app --host 0.0.0.0 --port 8080
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
