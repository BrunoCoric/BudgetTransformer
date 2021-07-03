FROM nvidia/cuda:10.2-base-ubuntu18.04

COPY requirements.txt ./
COPY main.py ./
COPY datasets.py ./
COPY utils.py ./
COPY nystrom_transformer.py ./
COPY losses.py ./
COPY engine.py ./
COPY models.py ./
COPY samplers.py ./
COPY hubconf.py ./

RUN apt-get update
RUN apt-get install nano
RUN apt-get -y install python3 python3-pip
RUN pip3 install --upgrade pip; \
    pip install -r requirements.txt

CMD python3 -u main.py