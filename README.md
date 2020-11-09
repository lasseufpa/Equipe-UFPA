# Petrobras-Challenge-CBR
Codes of LASSE team for "Petrobras Challenge of Robotics", which is part of the “CBR".

# Installation

First, you need to create a ROS workspace:

```bash
$ mkdir -p ~/lasse_ws/src
$ cd ~/lasse_ws/
$ catkin_make
$ source ~/lasse_ws/devel/setup.bash
$ echo 'source ~/lasse_ws/devel/setup.bash' >> ~/.bashrc 

# In the ~/.bashrc file, edit the following line to
# export ROS_WORKSPACES="~/mrs_workspace ~/workspace"
# export ROS_WORKSPACES="~/mrs_workspace ~/workspace ~/lasse_ws"
```

Then use the following commands to clone our ROS packages and repositories into the workspace:

PS: You will be asked credentials for gitlab, they are:

```bash
username: petrobraschallenge
password: ZNQCtuJ5wrpFN6
```

```bash
$ cd ~/lasse_ws/src
$ git config --global credential.helper store
$ git clone --recurse-submodules https://github.com/lasseufpa/Equipe-UFPA.git # You will be asked credentials here
```

After cloning do the following extra steps:

```bash
$ cd Equipe-UFPA/
$ mv hector_slam ~/workspace/src
$ cd ~/workspace/src
$ catkin build
$ cd ~/lasse_ws/src/Equipe-UFPA/vision_opencv
$ git checkout melodic
```

Now, build the workspace with:

```bash
$ cd ~/lasse_ws
$ catkin_make
```


To update submodules use (optional, can be done later):

```bash
$ cd ~/lasse_ws/src/Equipe-UFPA/<submodule>
$ git fetch
$ git merge
$ cd ~/lasse_ws
$ catkin_make
```

Install external dependecies:

```bash
$ sudo apt install python-pip python-opencv
$ sudo -H pip install numpy
$ sudo -H pip install pyzbar
```
