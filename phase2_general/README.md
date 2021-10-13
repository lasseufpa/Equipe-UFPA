# Description

This repository contains the files needed for phase 2.

# Installation

Go to your ROS workspace src/ directory.

```bash
$ git clone https://gitlab.lasse.ufpa.br/2020-petrobras-challenge/phase2_general.git
```

To use the scripts follow the additional steps bellow.

```bash
$ cd phase2_general/scripts

$ chmod +x *.py
```

Now you need to build the repository within your ROS workspace.

```bash
$ cd ~/<your ROS workspace>

$ catkin build

$ source devel/setup.bash
```

# Run

To run phase 2 use the following command.

```bash
$ rosrun phase2_general phase2_main.py
```
