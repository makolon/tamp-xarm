# TAMP-xARM
This repository implement PDDLStream for xArm7 and offer parallel reinforcement learning environment on Isaac Sim.

## Getting Started
### Prerequisited
- NVIDIA Docker
- NVIDIA RTX GPU
- NVIDIA Driver 515.xx

### Installation
1. Clone the repository
```
$ git clone --recursive git@github.com/makolon/tamp-xarm.git
```

2. Build docker image
```
$ cd tamp-xarm/docker/docker_xarm/
$ ./build.sh
```

3. Run docker container
```
$ cd tamp-xarm/docker/docker_xarm/
$ ./run.sh
```

4. Compile FastDownward
```
$ cd tamp-xarm/xarm_tamp/pddlstream/downward/
$ git submodule update --init --recursive
$ python3 build.py
$ cd ./builds/
$ ln -s release release32
```

## Usage
### Simulation
You can test PDDLStream on 3D pybullet environment including cooking, holding block task.
```
$ cd tamp-xarm/xarm_tamp/tampkit
$ python3 tamp_planner.py --problem <problem_name>
```

## Setup IKfast
### Compile IKfast
Build & run docker for openrave that contain IKfast scripts.
```
$ cd tamp-xarm/docker/docker_openrave/
$ ./build.sh
$ ./run.sh
```
Then, execute ikfast scripts that can automatically create cpp IK solver and copy and plaste the appropriate scripts to <ik_solver>.cpp.
```
$ ./exec_openrave.sh
```
After that process, you can call IK solver in python script by executing the following commands.
```
$ cd tamp-xarm/xarm_tamp/tampkit/sim_tools/pybullet/ikfast/xarm/
$ python3 setup.py
```

### Create xArm Collada Model
If you don't have xarm collada model, you have to run the following commands in docker_openrave container. \
Terminal 1.
```
$ cd /ikfast/
$ roscore
```
Terminal 2.
```
$ cd /ikfast
$ export MYROBOT_NAME='xarm'
$ rosrun collada_urdf urdf_to_collada "$MYROBOT_NAME".urdf "$MYROBOT_NAME".dae
```
Then, you can see the generated xArm collada model using following commands.
```
$ openrave-robot.py "$MYROBOT_NAME".dae --info links
$ openrave "$MYROBOT_NAME".dae
```

For more informations, please refer to the [following document](http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/ikfast/ikfast_tutorial.html).
