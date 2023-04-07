#!/usr/bin/env python

import numpy as np
import qlearn
import time

# ROS packages required
import rospy
import rospkg
import utils
from environment_stage_1_original import Env

if __name__ == '__main__':
    rospy.init_node('qlearn_training', anonymous=True, log_level=rospy.WARN)

    # Init environment
    max_step = rospy.get_param("/turtlebot3/nsteps")
    env = Env(action_dim=3, max_step=max_step)

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_rl_sim')
    result_outdir = pkg_path + '/src/results/qlearn'
    model_outdir = pkg_path + '/src/models/qlearn/discrete_no_greedy'

    # Remove log file if exist
    utils.remove_logfile_if_exist(result_outdir, "qlearn_training")

    # Loads parameters from the ROS param server
    Alpha = rospy.get_param("/turtlebot3/alpha")
    Epsilon = rospy.get_param("/turtlebot3/epsilon")
    Gamma = rospy.get_param("/turtlebot3/gamma")
    epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
    nepisodes = rospy.get_param("/turtlebot3/nepisodes")
    nsteps = rospy.get_param("/turtlebot3/nsteps")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(3), alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    # Discretization bins
    _distance_bins = [i for i in np.arange(0, 3, 0.1)] # 30 values max
    distance_bins = [round(i, 2) for i in _distance_bins]

    _radian_bins = [i for i in np.arange(-3.14, 3.14, 0.19625)] # 32 values max
    radian_bins = [round(i, 2) for i in _radian_bins]

    # Starts the main training loop: the one about the episodes to do
    for ep in range(nepisodes):
        cumulated_reward = 0
        env.done = False

        if qlearn.epsilon > 0.05:
            print("explore")
            qlearn.epsilon *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        time.sleep(0.1)
        env.done = False

        # Discretize observations
        _lsd_obs = [item for item in observation]  # Laser Scan Distances (m)
        _dtg_obs = [item for item in observation][-2]  # Distance To Goal (m)
        _htg_obs = [item for item in observation][-1]  # Heading To Goal (rad)

        discretized_lsd_obs = np.digitize(_lsd_obs, distance_bins)
        discretized_dtg_obs = np.digitize([_dtg_obs], distance_bins)
        discretized_htg_obs = np.digitize([_htg_obs], radian_bins)

        simple_continuous_obs = [_dtg_obs] + [_htg_obs]
        discretized_obs = np.concatenate([discretized_lsd_obs, discretized_dtg_obs, discretized_htg_obs])
        discretized_obs2 = np.concatenate([discretized_dtg_obs, discretized_htg_obs])

        state = ''.join(map(str, discretized_obs2))

        for step in range(nsteps):
            rospy.logwarn("EPISODE: " + str(ep+1) + " | STEP: " + str(step + 1))

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            observation, reward, done = env.step(action, step + 1)
            success_episode, failure_episode = env.get_episode_status()

            cumulated_reward += reward

            # Discretize observations
            _lsd_obs = [item for item in observation]  # Laser Scan Distances (m)
            _dtg_obs = [item for item in observation][-2]  # Distance To Goal (m)
            _htg_obs = [item for item in observation][-1]  # Heading To Goal (rad)

            discretized_lsd_obs = np.digitize(_lsd_obs, distance_bins)
            discretized_dtg_obs = np.digitize([_dtg_obs], distance_bins)
            discretized_htg_obs = np.digitize([_htg_obs], radian_bins)

            simple_continuous_obs = [_dtg_obs] + [_htg_obs]
            discretized_obs = np.concatenate([discretized_lsd_obs, discretized_dtg_obs, discretized_htg_obs])
            discretized_obs2 = np.concatenate([discretized_dtg_obs, discretized_htg_obs])

            nextState = ''.join(map(str, discretized_obs2))
            #qlearn.learn(state, action, reward, nextState)

            if not done:
                rospy.logwarn("NOT DONE")
                state = nextState
            if done:
                if (ep + 1) % 100 == 0:
                    # save Q-table
                    qtable = qlearn.get_qtable()
                    qlearn.save_q(qtable, model_outdir, "qlearn_qtable_ep" + str(ep+1))
                rospy.logwarn("DONE")
                data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1]
                utils.record_data(data, result_outdir, "qlearn_training_dis_no_greedy_test_3")
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                break

    env.reset()
