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

PS: You will be asked credentials for gitlab, they are:

```bash
username: ****
password: ****
```

```bash
$ cd ~/lasse_ws/src
$ mv CMakeLists.txt ../
$ git config --global credential.helper store
$ git clone --recurse-submodules https://github.com/lasseufpa/Petrobras-Challenge-CBR . # You will be asked credentials here
$ cd ~/lasse_ws
$ mv CMakeLists.txt ./src
```

After cloning do the following extra steps:

```bash
$ cd ~/lasse_ws/src/vision_opencv
$ git checkout kinetic
```

Now, build the workspace with:

```bash
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