FROM gcr.io/kaggle-gpu-images/python:v129
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cudagrad
COPY . /cudagrad
RUN pip install -r dev-requirements.txt
RUN rm -rf dist
RUN python project.py build
