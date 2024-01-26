FROM nvcr.io/nvidia/isaac-sim:2023.1.1

ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    bash-completion \
    build-essential \
    cmake \
    curl \
    dbus-x11 \
    gdb \
    gedit \
    git \
    gnupg2 \
    htop \
    libssl-dev \
    locales \
    lsb-release \
    net-tools \
    python3-pip \
    software-properties-common \
    subversion \
    sudo \
    terminator \
    unzip \
    valgrind \
    vim \
    wget \
    xterm && \
    apt-get clean && rm -rf /var/lib/apt/list*

# Setup Isaac Sim
ENV ISAACSIM_PATH="${HOME}/isaac-sim"
ENV ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

RUN echo "alias PYTHON_PATH=/isaac-sim/python.sh" >> ${HOME}/.bashrc
RUN echo "export DISPLAY=:1" >> ${HOME}/.bashrc
RUN echo "export PYTHONPATH=$PYTHONPATH:/root/tamp-xarm" >> ${HOME}/.bashrc

# Setup Python Package
RUN apt-get update
RUN ${ISAACSIM_PYTHON_EXE} -m pip install -U pip
RUN ${ISAACSIM_PYTHON_EXE} -m pip install \
    absl-py==1.2.0 \
    aiosignal==1.2.0 \
    antlr4-python3-runtime \
    cachetools==4.2.4 \
    cloudpickle==2.1.0 \
    cycler==0.11.0 \
    cython==0.29.32 \
    decorator==4.4.2 \
    distlib==0.3.5 \
    docker-pycreds==0.4.0 \
    eigenpy==3.3.0 \
    filelock==3.7.1 \
    fonttools==4.34.4 \
    frozenlist==1.3.0 \
    gitdb==4.0.9 \
    gitpython==3.1.27 \
    google-auth-oauthlib==0.4.6 \
    google-auth==1.35.0 \
    grpcio \
    hpp-fcl==2.4.1 \
    hydra-core==1.3.2 \ 
    imageio-ffmpeg==0.4.7 \
    imageio==2.21.1 \
    importlib-metadata==4.12.0 \
    importlib-resources==5.9.0 \
    iniconfig==1.1.1 \
    joblib==1.1.0 \
    jupyterlab==3.6.6 \
    kiwisolver==1.4.4 \
    markdown==3.4.1 \
    matplotlib==3.5.2 \
    moviepy==1.0.3 \
    msgpack==1.0.4 \
    numpy \
    numpy-ml==0.1.2 \
    oauthlib==3.2.0 \
    omegaconf==2.3.0 \
    opencv-python==4.6.0.66 \
    packaging==21.3 \
    pandas==1.3.5 \
    pin==2.7.0 \
    platformdirs==2.5.2 \
    pluggy==1.0.0 \
    proglog==0.1.10 \
    promise==2.3 \
    protobuf==3.20.2 \
    py==1.11.0 \
    pyasn1-modules==0.2.8 \
    pyasn1==0.4.8 \
    pybullet==3.2.5 \
    pygame==2.1.2 \
    pyparsing==3.0.9 \
    pytest==7.1.2 \
    python-dateutil==2.8.2 \
    pytz==2022.1 \
    qpsolvers==2.2.0 \
    quadprog==0.1.11 \
    ray==1.13.0 \
    requests-oauthlib==1.3.1 \
    rl-games==1.6.1 \
    rsa==4.9 \
    scikit-learn==1.0.2 \
    scipy \
    sentry-sdk==1.9.5 \
    setproctitle==1.2.3 \
    setuptools==65.6.0 \
    shapely==1.8.5 \
    shortuuid==1.0.9 \
    smmap==5.0.0 \
    stable-baselines3==1.2.0 \
    threadpoolctl==3.1.0 \
    tomli==2.0.1 \
    torch==1.13.0 \
    torchvision==0.14.0 \
    tqdm==4.64.0 \
    virtualenv==20.15.1 \
    wandb \
    werkzeug==2.1.2 \
    zipp==3.8.1

# Setup Workdir
WORKDIR /root

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash"]

