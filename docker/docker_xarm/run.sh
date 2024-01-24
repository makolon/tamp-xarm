#!/bin/bash
docker run -it --rm --privileged --gpus all --net host --ipc host \
	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ~/tamp-xarm:/root/tamp-xarm \
    --name xarm_tamp xarm_tamp:latest bash