FROM continuumio/miniconda3@sha256:145567896379c33c89737aa1ab1389a71aead35b39f512118a13edcf9fe2a42c

COPY . /app

WORKDIR /app

RUN conda env create -f environment.yml

EXPOSE 5000
CMD conda run -n autochemplete-ml python app.py --device cpu
