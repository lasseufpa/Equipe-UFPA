# Description

Package that offers a python module that contains a class to retrieve images from the camera.

# Dependecies

- OpenCV (Already installed in the VM/It comes with the full install of MRS system (I think))

Install in your workspace:

- https://github.com/ros-perception/vision_opencv.git (branch kinetic)
- https://gitlab.lasse.ufpa.br/2020-petrobras-challenge/phase0_control (lastest version)


# How to use

After build the package in your workspace, import the class `GetImage` with:

```python
from phase0_drone_camera.camera import GetImage
```

Use the method `.get()` to retrive images (in opencv format)
