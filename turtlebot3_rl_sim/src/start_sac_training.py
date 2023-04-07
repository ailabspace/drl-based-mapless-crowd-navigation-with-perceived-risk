#!/usr/bin/env python

"""
Based on:
https://github.com/dranaju/project
"""
import sac
import rospy
import numpy as np
import rospkg
import utils
import time
from environment_stage_1_original import Env

# Importing the library
import psutil

if __name__ == '__main__':
    rospy.init_node('sac_training', anonymous=True, log_level=rospy.WARN)

    # Init environment
    max_step = rospy.get_param("/turtlebot3/nsteps")
    env = Env(action_dim=2, max_step=max_step)
    stage_name = rospy.get_param("/turtlebot3/stage_name")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_rl_sim')
    result_outdir = pkg_path + '/src/results/sac' + '/' + stage_name
    model_outdir = pkg_path + '/src/models/sac' + '/' + stage_name
    actor_model_param_path = model_outdir + '/sac_actor_model_ep'
    critic_v_model_param_path = model_outdir + '/sac_critic_v_model_ep'
    critic_soft_q_model_param_path = model_outdir + '/sac_critic_soft_q_model_ep'

    # Remove log file if exist
    # utils.remove_logfile_if_exist(result_outdir, "td3_training")

    resume_epoch = 600
    continue_execution = False
    actor_resume_path = actor_model_param_path + str(resume_epoch)
    critic_v_resume_path = critic_v_model_param_path + str(resume_epoch)
    critic_soft_q_resume_path = critic_soft_q_model_param_path + str(resume_epoch)
    actor_path = actor_resume_path + '.pt'
    critic_v_path = critic_v_resume_path + '.pt'
    critic_soft_q_path = critic_soft_q_resume_path + '.pt'

    if not continue_execution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_v_learning_rate = rospy.get_param("/turtlebot3/critic_v_alpha")
        critic_soft_q_learning_rate = rospy.get_param("/turtlebot3/critic_soft_q_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 64  # 128
        memory_size = 1000000
        network_inputs = 54  # State dimension
        hidden_layers = 256  # Hidden dimension
        network_outputs = 2  # Action dimension
        action_v_max = 0.22  # m/s
        action_w_max = 2.0  # rad/s
        resume_epoch = 0
        mean_lambda = 0.001
        std_lambda = 0.001
        z_lambda = 0.0

        sac_trainer = sac.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                critic_v_learning_rate, critic_soft_q_learning_rate, batch_size, memory_size,
                                discount_factor, softupdate_coefficient, action_v_max, action_w_max, mean_lambda,
                                std_lambda, z_lambda)

    else:
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_v_learning_rate = rospy.get_param("/turtlebot3/critic_v_alpha")
        critic_soft_q_learning_rate = rospy.get_param("/turtlebot3/critic_soft_q_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 64  # 128
        memory_size = 1000000
        network_inputs = 54  # State dimension
        hidden_layers = 256  # Hidden dimension
        network_outputs = 2  # Action dimension
        action_v_max = 0.22  # m/s
        action_w_max = 2.0  # rad/s
        resume_epoch = 600
        mean_lambda = 0.001
        std_lambda = 0.001
        z_lambda = 0.0

        sac_trainer = sac.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                critic_v_learning_rate, critic_soft_q_learning_rate, batch_size, memory_size,
                                discount_factor, softupdate_coefficient, action_v_max, action_w_max, mean_lambda,
                                std_lambda, z_lambda)
        sac_trainer.load_models(actor_path, critic_soft_q_path, critic_v_path)

    for ep in range(resume_epoch, nepisodes):
        cumulated_reward = 0

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        time.sleep(0.1)  # Give time for RL to reset the agent's position
        env.done = False
        state = observation

        for step in range(nsteps):
            # Getting % usage of virtual_memory ( 3rd field)
            print('RAM memory % used:', psutil.virtual_memory()[2])
            rospy.logwarn("EPISODE: " + str(ep + 1) + " | STEP: " + str(step + 1))

            state = np.float32(state)
            action = sac_trainer.act(state)
            _action = action.flatten().tolist()
            observation, reward, done = env.step(_action, step + 1, mode="continuous")
            success_episode, failure_episode = env.get_episode_status()
            cumulated_reward += reward

            next_state = observation
            next_state = np.float32(next_state)

            # Learning
            sac_trainer.memory.add(state, action, reward, next_state, done)
            if len(sac_trainer.memory) > batch_size:
                sac_trainer.learn()

            if not done:
                rospy.logwarn("NOT DONE")
                state = next_state

            if done:
                # Debugging purposes
                if (step + 1) <= 2:
                    env.shutdown()
                    raw_input("Press Enter to continue...")
                if (ep + 1) % 100 == 0:
                    # save model weights and monitoring data every 100 epochs.
                    sac_trainer.save_actor_model(model_outdir, "sac_actor_model_ep" + str(ep + 1) + '.pt')
                    sac_trainer.save_critic_v_model(model_outdir, "sac_critic_v_model_ep" + str(ep + 1) + '.pt')
                    sac_trainer.save_critic_soft_q_model(model_outdir, "sac_critic_soft_q_model_ep" + str(ep + 1) + '.pt')
                rospy.logwarn("DONE")
                data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1]
                utils.record_data(data, result_outdir, "sac_training")
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                break

    env.reset()
