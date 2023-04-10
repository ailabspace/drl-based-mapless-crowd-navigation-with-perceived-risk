# Deep Reinforcement Learning Based Crowd Navigation with Perceived Risk of the Moving Crowd for Mobile Robots

Source code for research work on adding Risk Perception to TD3 for Crowd Navigation. It also provides a framework to train and test six different algorithms which are TD3, DDPG, SAC, Q-Learning, SARSA, and DQN.


[![Watch the video](https://img.youtube.com/vi/djD9mfPQRgc/maxresdefault.jpg)](https://youtu.be/djD9mfPQRgc)

If you have found this repository useful or have used this repository in any of your scientific work, please consider citing my work using this [BibTeX Citation](#bibtex-citation). A full demonstration video of the mobile robot navigation has been uploaded on [Youtube](https://www.youtube.com/watch?v=djD9mfPQRgc&t=1s).

## Table of contents

* [Installation](#installation)
* [Repository contents](#repository-contents)
* [Getting started](#getting-started)
* [Hardware and software information](#hardware-and-software-information)
* [BibTeX Citation](#bibtex-citation)
* [Acknowledgments](#acknowledgments)

## Installation

- Firstly, the following packages ([turtlebot3](http://wiki.ros.org/turtlebot3), [turtlebot3_gazebo](http://wiki.ros.org/turtlebot3_gazebo)) and their dependencies should be cloned in your ROS workspace.
- Then, clone this repository and move the contents turtlebot3_simulations and turtlebot3_description to the installed packages.
- Finally, the ROS workspace should be compiled with `catkin_make` and sourced with `source devel/setup.bash`. The compile process should return no error if all the dependencies are met. 

## Repository contents

**turtlebot3_rl_sim** - This folder contains files for the robot to run our version of TD3 (with Risk Perception of Crowd) as well as other algorithms of DDPG, TD3, DQN, Q-Learning, and SARSA for training and testing.

**turtlebot3_description** - This folder contains core files to run Turtlebot3 in Gazebo simulator with the same settings used in our work.

**turtlebot3_simulations** - This folder contains the Gazebo simulation launch files, models, and worlds.

## Getting Started

**Start ROSCORE**

1. Run `roscore` in your terminal.
 

**Launch Gazebo world**

2. Run `roslaunch turtlebot3_gazebo turtlebot3_crowd_dense.launch` in your terminal.

**Place your robot in the Gazebo world** 

3. Run `roslaunch turtlebot3_gazebo put_robot_in_world.launch` in your terminal.

**Simulating crowd behavior** 

4. Run `rosrun turtlebot3_rl_sim simulate_crowd.py` in your terminal.

**Start training with TD3** 

5. Run `roslaunch turtlebot3_rl_sim start_td3_training.launch` in your terminal.

**Real world testing (deployment)** 

6. Physical deployment requires the Turtlebot3 itself and a remote PC to run.

7. On the Turtlebot3:
- Run `roslaunch turtlebot3_bringup turtlebot3_robot.launch`

8. On the remote PC:
- Run `roscore`
- Run `roslaunch turtlebot3_bringup turtlebot3_remote.launch`
- Run `roslaunch turtlebot3_rl_sim start_td3_real_world_test.launch`

## Hardware and Software Information

**Software**

- OS: Ubuntu 18.04
- ROS version: Melodic
- Python version: 2.7
- Gazebo version: 9
- CUDA version: 10.0
- CuDNN version: 7

**Computer Specifications**

- CPU: Intel i7 9700
- GPU: Nvidia RTX 2070

**Mobile Robot Platform**

- [Turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

## BibTeX Citation

If you have used this repository in any of your scientific work, please consider citing the work (Submitted to IROS2023):

```
@misc{anas2023deep,
      title={Deep Reinforcement Learning-Based Mapless Crowd Navigation with Perceived Risk of the Moving Crowd for Mobile Robots}, 
      author={Hafiq Anas and Ong Wee Hong and Owais Ahmed Malik},
      year={2023},
      eprint={2304.03593},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgments

* Thank you [Robolab@UBD](https://ailab.space/) for lending the Turtlebot3 robot platform and lab facilities.

