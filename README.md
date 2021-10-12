# Petrobras-Challenge-CBR
Codes of LASSE team for "Petrobras Challenge of Robotics", which is part of the “CBR".

# Dependencies

- OpenCV 
- Pyzbar
- vision_opencv
- torch
- torchvision
- scikit-build

# Installation

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

rosrun phase0_base_centralize node

rosrun phase1_panoramic phase1_land.py

rosrun phase2_general phase2_script.py

-----



