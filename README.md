# Petrobras-Challenge-CBR
Codes of LASSE team for "Petrobras Challenge of Robotics", which is part of the “CBR".

# Dependencies

- Hector SLAM (MRS Fork)
- OpenCV (probably not needed)
- Pyzbar
- vision_opencv
- gazebo_ros_link_attacher

# Installation

Use the following commands to clone our ROS packages and repositories into the workspace:

PS: You will be asked credentials for gitlab, they are:

```bash
username: petrobraschallenge
password: ZNQCtuJ5wrpFN6
```

```bash
$ cd ~/workspace
$ git config --global credential.helper store
$ git clone --recurse-submodules https://github.com/lasseufpa/Equipe-UFPA.git # You will be asked credentials here
```

After cloning do the following extra steps:

Now, build the workspace with:

```bash
$ cd ~/workspace
$ catkin build
$ source ~/workspace/devel/setup.bash
```

## Update

To update submodules use (optional, can be done later):

```bash
$ cd ~/lasse_ws/src/Equipe-UFPA/<submodule>
$ git fetch
$ git merge
$ cd ~/lasse_ws
$ catkin_make
$ git push
```

Or, if you want to update all submodules (it will use the master branch, be careful)

```bash
$ git submodule update --remote
$ git push
```

## External dependecies:

```bash
$ sudo apt install python-pip python-opencv
$ sudo -H pip install numpy
$ sudo -H pip install pyzbar
```
