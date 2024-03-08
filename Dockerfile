FROM python:3.8.6-buster
WORKDIR /prod

RUN pip install --upgrade pip

COPY requirements_dev.txt requirements.txt
RUN pip install -r requirements.txt

COPY crops_package crops_package
COPY setup.py setup.py
RUN pip install .

COPY Makefile Makefile
# RUN make reset_local_files

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
