FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

WORKDIR /root/

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iproute2 \
        g++-7 \
        gcc-7 \
        wget \
        vim \
        git \
        curl \
        gnupg \
        xvfb \
        terminator \
        build-essential \
        openssh-client \
        net-tools \
        lsb-release \
        tk-dev \
        xorg-dev \
        software-properties-common \
        libglu1-mesa-dev \
        libsdl2-dev \
        libc++-7-dev \
        libc++abi-7-dev \
        ninja-build \
        libxi-dev \
        libtbb-dev \
        libosmesa6-dev \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get install -y build-essential
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    && rm -rf /var/lib/apt/lists/*

### Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        python3-pyqt5 \
        python3-pip \
        python3-catkin-pkg \
        python3-catkin-tools \
        python3-catkin-pkg-modules \
        python3-rospkg-modules \
        python3-opencv \
        python3-empy \
        python3-dev \
        python3-tk \
        ros-noetic-tf* \
        ros-noetic-ros-numpy* \
    && rm -rf /var/lib/apt/lists/*

### Install python packages
RUN pip3 install -U pip
RUN pip3 install --ignore-installed \
    setuptools \
	gym \
	numpy \
	pybullet \
	matplotlib \
	opencv-python \
	scipy \
	scikit-image \
	tensorflow \
	tensorflow-addons \
	tensorflow_hub\
	transforms3d \
	keras \
	jupyterlab \
    torch \
    torchvision \
    kinpy \
    pin \
    trimesh \
    pygame \
    catkin_tools \
    rl-games \
    warp-lang

### Install ROS packages
RUN mkdir -p /root/catkin_ws/src \
    && cd /root/catkin_ws/ \
    && rosdep init \
    && rosdep update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-noetic-joint-state-publisher-gui \
        ros-noetic-ros-control \
        ros-noetic-ros-controllers \
        ros-noetic-gazebo-ros-control \
        ros-noetic-moveit \
        ros-noetic-moveit-ros-visualization \
        ros-noetic-mocap-optitrack \
    && rm -rf /var/lib/apt/lists/*

### System settings
ENV ROS_PACKAGE_PATH=/root/catkin_ws:$ROS_PACKAGE_PATH
ENV ROS_WORKSPACE=/root/catkin_ws
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "export DISPLAY=:0" >> ~/.bashrc
RUN echo "export PYTHONPATH='/root/tamp-xarm/xarm_tamp:$PYTHONPATH'" >> ~/.bashrc

### Install CUDA
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb && rm cuda-keyring_1.0-1_all.deb
