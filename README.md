# TD3, DDPG, SAC, Q-Learning, SARSA, and DQN Reinforcement Learning-based mobile robot navigation

This repository contains codes to replicate my research work titled "**Deep Reinforcement Learning-Based Mapless Crowd Navigation with Perceived Risk of the Moving Crowd for Mobile Robots**". 

In addition, it also provides a framework to train and test six different algorithms which are TD3, DDPG, SAC, Q-Learning, SARSA, and DQN. Turtlebot3 Burger mobile robot platform was used to train and test these algorithms. Unlike my [other repository](https://github.com/zerosansan/dqn_qlearning_sarsa_mobile_robot_navigation), I have completely removed the dependency of OpenAI so it is a much simpler process to get our codes to run in your own workspace.

[![Watch the video](https://img.youtube.com/vi/djD9mfPQRgc/maxresdefault.jpg)](https://youtu.be/djD9mfPQRgc)

If you have found this repository useful or have used this repository in any of your scientific work, please consider citing my work using this [BibTeX Citation](#bibtex-citation). A full mobile robot navigation demonstration video has been uploaded on [YouTube](https://www.youtube.com/watch?v=djD9mfPQRgc&t=1s).

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

**turtlebot3_description** - This folder contains core files to run Turtlebot3 in the Gazebo simulator with the same settings used in our work.

**turtlebot3_simulations** - This folder contains the Gazebo simulation launch files, models, and worlds.

## Getting Started

**Start ROSCORE**

1. Run `roscore` in your terminal.

**Launch Gazebo world**

2. Run `roslaunch turtlebot3_gazebo turtlebot3_crowd_dense.launch` in your terminal.

**Place your robot in the Gazebo world** 

3. Run `roslaunch turtlebot3_gazebo put_robot_in_world_training.launch` in your terminal.

**Simulating crowd behavior** 

4. Run `rosrun turtlebot3_rl_sim simulate_crowd.py` in your terminal.

**Start training with TD3** 

5. Run `roslaunch turtlebot3_rl_sim start_td3_training.launch` in your terminal.

**Start testing with TD3**

Firstly, we must use the following parameters in the `start_td3_training.py` script:
```python
resume_epoch = 1500  # e.g. 1500 means it will use the model saved at episode 1500
continue_execution = True
learning = False
k_obstacle_count = 8  # K = 8 implementation
utils.record_data(data, result_outdir, "td3_training_trajectory_test") <-- Change the string name accordingly, to avoid overwriting the training results file
```
Secondly, edit the `turtlebot3_world.yaml` file to reflect the following settings:
```yaml
min_scan_range: 0.0 # To get reliable social and ego score readings, depending on evaluation metrics
desired_pose:
    x: -2.0
    y: 2.0
    z: 0.0
starting_pose:
    x: 1.0
    y: 0.0
    z: 0.0
```
Thirdly, edit the `environment_stage_1_nobonus.py` script to reflect the following settings:
```python
self.k_obstacle_count = 8  #K = 8 implementation
```
1. Launch the Gazebo world:
- Run `roslaunch turtlebot3_gazebo turtlebot3_obstacle_20.launch` in your terminal.
2. Place your robot in the Gazebo world:
- Run `roslaunch turtlebot3_gazebo put_robot_in_world_testing.launch` in your terminal.
3. Simulating test crowd behaviors (OPTIONS:{crossing, towards, ahead, random}):
- Run `rosrun turtlebot3_rl_sim simulate_OPTIONS_20.py` in your terminal.
4. Start the testing script:
- Run `roslaunch turtlebot3_rl_sim start_td3_training.launch` in your terminal.

**Real-world testing (deployment)** 

1. Physical deployment requires the Turtlebot3 itself and a remote PC to run.

2. On the Turtlebot3:
- Run `roslaunch turtlebot3_bringup turtlebot3_robot.launch` in your terminal.

3. On the remote PC:
- Run `roscore`
- Run `roslaunch turtlebot3_bringup turtlebot3_remote.launch` in your terminal.
- Run `roslaunch turtlebot3_rl_sim start_td3_real_world_test.launch` in your terminal.

## Hardware and Software Information

**Software**

- OS: Ubuntu 18.04
- ROS version: Melodic
- Python version: 2.7 (Code is in Python3, so porting to a newer version of ROS/Ubuntu should have no issues)
- Gazebo version: 9.19
- CUDA version: 10.0
- CuDNN version: 7

**Computer Specifications**

- CPU: Intel i7 9700
- GPU: Nvidia RTX 2070

**Mobile Robot Platform**

- [Turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

## BibTeX Citation

If you have used this repository in any of your scientific work, please consider citing my work (submitted to ICRA2024):

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

