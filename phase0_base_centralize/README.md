# Description

ROS node that offers a service to check if the drone is centralized in a base or not. It also give directions to centralize.

# Dependecies

- OpenCV (Already installed in the VM/It comes with the full install of MRS system (I think))

Install in your workspace:

- https://github.com/ros-perception/vision_opencv.git

# How to use

After build the package in your workspace, run the ROS node that offers the service with:

```bash
$ rosrun phase0_base_centralize node
```

# Services

## /base_centralize

It returns the offset to where move the drone.

Check for more info `rosservice info /base_centralize`

Check for more info  `rossrv info phase0_base_centralize/GetOffset`
