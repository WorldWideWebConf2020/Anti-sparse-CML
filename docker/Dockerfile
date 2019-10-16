FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# basic setup
RUN apt-get update
RUN apt-get install nano

# install anaconda
ENV PATH /opt/conda/bin:$PATH
COPY install_miniconda.sh .
RUN bash ./install_miniconda.sh && rm install_miniconda.sh

# c compiler
#RUN sudo apt install -y gcc

# ML machinery
RUN conda update conda
RUN conda install -c anaconda tensorflow-gpu
RUN conda install -c anaconda pandas
RUN conda install -c anaconda line_profiler
RUN pip install ray
RUN conda install -c anaconda psutil 

WORKDIR /workspace/DREC
ENV PYTHONPATH /workspace/DREC
ENV CUDA_VISIBLE_DEVICES 0
ENV CONDA_PREFIX /opt/conda
