FROM gcr.io/kaggle-images/python:latest
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /cudagrad
COPY . /cudagrad
RUN pip install -r dev-requirements.txt
RUN rm -rf dist
RUN python project.py build
