# Petrobras-Challenge-CBR
Codes of LASSE team for "Petrobras Challenge of Robotics", which is part of the “CBR".

# Installation

First, you need to create a ROS workspace:

```bash
$ mkdir -p ~/lasse_ws/src
$ cd ~/lasse_ws/
$ catkin_make
```

Then use the following commands to clone our ROS packages and repositories into the workspace:

```bash
$ cd ~/lasse_ws/src
$ git clone --recurse-submodules https://github.com/lasseufpa/Petrobras-Challenge-CBR .
$ cd ~/lasse_ws
$ catkin_make
```

To update submodules use (optional, can be done later):

```bash
$ cd ~/lasse_ws/src
$ git submodule update --remote
$ cd ~/lasse_ws
$ catkin_make
```

Install external dependecies:

```bash
$ sudo apt install python-pip python-opencv
$ sudo pip install numpy
```