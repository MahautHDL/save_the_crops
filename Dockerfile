FROM python:3.10.6-buster

WORKDIR /prod

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY crops_package crops_package
COPY api api
COPY setup.py setup.py

CMD uvicorn api.fast:app --host 0.0.0.0 --port 8000
