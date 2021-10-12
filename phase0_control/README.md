# Description

Este repositorio contem scripts e classes para o controle do drone.

Os scripts estão dentro de scripts/ e devem ser rodados usando `rosrun phase0_control <script>`.

As classes estao em arquivos dentro de src/ e devem ser usadas importando para o seu codigo em python usando `from phase0_control.<src file name> import <class name>`.

# Installation


Go to your ROS workspace src/ directory.

```bash
$ git clone https://gitlab.lasse.ufpa.br/2020-petrobras-challenge/phase0_control_gps.git

$ cd phase0_control_gps 
 
```

If you want to use the scripts follow the additional steps bellow.

```bash
$ cd phase0_control_gps/scripts

$ chmod +x *.py
 
```

Now you need to build the repository within your ROS worspace

PS: If the bellow command fails, try to use `catkin_make` instead.

```bash
$ cd ~/<your ROS workspace>

$ catkin build

$ source devel/setup.bash
```

# Scripts

## control.py

`rosrun phase0_control control.py`

Read a trace file and make the drone follow it.

You can check the format of the trace in `data/test_trace.json`. Every array `[x , y, z]` is an absolute position to the drone go to.

The program still lacks a feature to check if the drone arrived at the destination, so between every displacement the drone has 10 seconds to arrive.

# Classes

For every class, you need to pay attetion to public methods and the constructor.

Private methods follow the convention of having a  `_` before its name, so you want to take a look in methods that do not have it (public methods).

The constructor is the  `__init__` method.

## Controller (inside controller.py)

Class to control the drone using absolute or relative position commands.

`change_reference_pos` method

