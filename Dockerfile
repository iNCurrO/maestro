FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update 
RUN apt-get install -y -q openssh-server

RUN apt-get install -y -q \
    wget \
    zlib1g-dev \
    curl \
    libssl-dev \
    libffi-dev \
    vim \
    libgl1-mesa-glx \
    zip \
    libglib2.0-0 \
    git
    
RUN apt-get install -y -q \
    libsqlite3-dev


# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python && pip install -U pip

RUN pip install pydicom matplotlib scipy gdown opencv-python vessl h5py mat73 pyspng
RUN pip install timm==0.9.2

RUN pip install jupyterlab
RUN ln -s /opt/conda/bin/jupyter /usr/local/bin/jupyter