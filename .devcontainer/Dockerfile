FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

COPY ./requirements.txt /requirements.txt
RUN pip3 -q install pip --upgrade
RUN pip3 install -r /requirements.txt
RUN apt-get update && apt-get install -y \
        tmux \ 
        git \
        wget \
        unzip \
        gnupg2
