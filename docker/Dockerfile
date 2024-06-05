FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
# FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt update -qq && \
    apt install -y git vim libgtk2.0-dev

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt
