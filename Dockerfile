FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    software-properties-common \
    cmake \
    python3-pip \
    python3-setuptools \
    python3-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    && apt-get clean
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar xzf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations --enable-shared \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.10.14 Python-3.10.14.tgz
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/python3.10.conf && ldconfig
RUN update-alternatives --install /usr/local/bin/python python /usr/local/bin/python3.10 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3.10 1
WORKDIR /cudagrad
COPY . /cudagrad
RUN pip install -r dev-requirements.txt
ENTRYPOINT ["/bin/bash"]
