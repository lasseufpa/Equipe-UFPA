# Petrobras-Challenge-CBR
Codes of LASSE team for "Petrobras Challenge of Robotics", which is part of the “CBR".

# Dependencies

- Hector SLAM (MRS Fork)
- OpenCV (probably not needed)
- Pyzbar
- vision_opencv
- gazebo_ros_link_attacher
- torch
- torchvision
- scikit-build

# Installation

Use the following commands to clone our ROS packages and repositories into the workspace:

PS: You will be asked credentials for **gitlab**, they are:

```bash
username: petrobraschallenge
password: ZNQCtuJ5wrpFN6
```

```bash
$ cd ~/workspace/src
$ git config --global credential.helper store
$ git clone --recurse-submodules https://github.com/lasseufpa/Equipe-UFPA.git # You will be asked credentials here
```

After cloning do the following extra steps:

Now, build the workspace with:

```bash
$ cd ~/workspace
$ catkin build
$ source ~/workspace/devel/setup.bash
$ echo 'source ~/workspace/devel/setup.bash' >> ~/.bashrc 
```

## Update

To update all submodules (it will use the master branch, be careful)

```bash
$ git submodule update --remote
```

## External dependecies:

```bash
$ sudo apt install python-pip python-opencv
$ sudo -H pip install numpy
$ sudo -H pip install pyzbar
$ sudo -H pip install torch
$ sudo -H pip install torchvision
$ sudo -H pip install scikit-build
```
# Cuda:

## Driver (apenas se não tiver o driver Nvidia):

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-450
```

## Cuda:

```
sudo dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb`
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub`
sudo apt-get update`
sudo apt-get install cuda`
```

Ou: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal

# Testing phases 1 and 2:

rosrun phase1_panoramic main.py

rosrun phase2_general phase2_script.py

-----

rosrun phase1_base_centralize node

rosrun phase1_general phase1.py --maps <path>/Equipe-UFPA/phase1_explore/data/


