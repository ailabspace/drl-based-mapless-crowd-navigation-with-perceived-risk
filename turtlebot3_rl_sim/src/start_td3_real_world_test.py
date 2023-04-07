#!/usr/bin/env python

"""
Based on:
https://github.com/dranaju/project
"""
import td3
import rospy
import numpy as np
import rospkg
import utils
import time
from environment_stage_1_nobonus_realworld import Env

# Importing the library
import psutil
import timeit

if __name__ == '__main__':
    rospy.init_node('td3_training', anonymous=True)

    # Init environment
    max_step = rospy.get_param("/turtlebot3/nsteps")
    env = Env(action_dim=2, max_step=max_step)
    stage_name = rospy.get_param("/turtlebot3/stage_name")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_rl_sim')
    result_outdir = pkg_path + '/src/results/td3_final' + '/' + stage_name
    model_outdir = pkg_path + '/src/models/td3_final' + '/' + stage_name
    actor_model_param_path = model_outdir + '/td3_actor_model_ep'
    critic1_model_param_path = model_outdir + '/td3_critic1_model_ep'
    critic2_model_param_path = model_outdir + '/td3_critic2_model_ep'

    # Remove log file if exist
    # utils.remove_logfile_if_exist(result_outdir, "td3_training")

    resume_epoch = 3000
    continue_execution = True
    actor_resume_path = actor_model_param_path + str(resume_epoch)
    critic1_resume_path = critic1_model_param_path + str(resume_epoch)
    critic2_resume_path = critic2_model_param_path + str(resume_epoch)
    actor_path = actor_resume_path + '.pt'
    critic1_path = critic1_resume_path + '.pt'
    critic2_path = critic2_resume_path + '.pt'

    if not continue_execution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_learning_rate = rospy.get_param("/turtlebot3/critic_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 128  # 128
        memory_size = 1000000
        network_inputs = 370 #74 #38 #54  # State dimension
        hidden_layers = 256  # Hidden dimension
        network_outputs = 2  # Action dimension
        action_v_max = 0.22 #0.22  # m/s
        action_w_max = 2.0 #2.0  # rad/s
        resume_epoch = 0
        noise_std = 0.2
        noise_clip = 0.5
        policy_update = 2

        td3_trainer = td3.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                critic_learning_rate, batch_size, memory_size, discount_factor,
                                softupdate_coefficient, action_v_max, action_w_max, noise_std, noise_clip,
                                policy_update)

    else:
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        actor_learning_rate = rospy.get_param("/turtlebot3/actor_alpha")
        critic_learning_rate = rospy.get_param("/turtlebot3/critic_alpha")
        discount_factor = rospy.get_param("/turtlebot3/gamma")
        softupdate_coefficient = rospy.get_param("/turtlebot3/tau")
        batch_size = 256  # 128
        memory_size = 1000000
        network_inputs = 370 #74  # State dimension
        hidden_layers = 256  # Hidden dimension
        network_outputs = 2  # Action dimension
        action_v_max = 0.22  # m/s
        action_w_max = 2.0  # rad/s
        # resume_epoch = 100
        noise_std = 0.2
        noise_clip = 0.5
        policy_update = 2

        td3_trainer = td3.Agent(network_inputs, network_outputs, hidden_layers, actor_learning_rate,
                                critic_learning_rate, batch_size, memory_size, discount_factor,
                                softupdate_coefficient, action_v_max, action_w_max, noise_std, noise_clip,
                                policy_update)
        td3_trainer.load_models(actor_path, critic1_path, critic2_path)

    step_counter = 0
    time_lapse = 0
    for ep in range(resume_epoch, nepisodes):
        rospy.logwarn("EPISODE: " + str(ep + 1))
        cumulated_reward = 0
        social_safety_score = 0
        ego_safety_score = 0

        # Initialize the environment and get first state of the robot
        # observation = env.reset()
        print("START OBSERVATION")
        observation = ([0.6] * 360) + [0.0, 1.6] + [0.0, 0.0] + [3.14] + [0.0] + [0.0, 0.0] + [0.0, 0.0]
        print(len(observation))
        print("END OBSERVATION")
        time.sleep(0.1)  # Give time for RL to reset the agent's position
        start_time = time.time()
        env.done = False
        state = observation

        for step in range(nsteps):
            # Getting % usage of virtual_memory ( 3rd field)
            # print('RAM memory % used:', psutil.virtual_memory()[2])
            rospy.logwarn("EPISODE: " + str(ep + 1) + " | STEP: " + str(step + 1))
            step_counter += 1
            # rospy.logwarn("STEP counter: " + str(step_counter))
            state = np.float32(state)
            action = td3_trainer.act(state, step, add_noise=False)
            _action = action.flatten().tolist()
            observation, reward, done = env.step(_action, step + 1, mode="continuous")
            success_episode, failure_episode = env.get_episode_status()
            cumulated_reward += reward

            next_state = observation
            next_state = np.float32(next_state)

            # Learning
            # td3_trainer.memory.add(state, action, reward, next_state, done)
            # if len(td3_trainer.memory) > batch_size:
            #     td3_trainer.learn(step)

            if not done:
                # rospy.logwarn("NOT DONE"
                # print("HTG OBSERVATION: ", state[359])
                state = next_state

            if done:
                time_lapse = time.time() - start_time
                social_safety_score = env.get_social_safety_violation_status(step + 1)
                ego_safety_score = env.get_ego_safety_violation_status(step + 1)
                # Debugging purposes
                if (step + 1) <= 2:
                    env.shutdown()
                    # raw_input("Press Enter to continue...")
                if (ep + 1) % 100 == 0:
                    # save model weights and monitoring data every 100 epochs.
                    td3_trainer.save_actor_model(model_outdir, "td3_actor_model_ep" + str(ep + 1) + '.pt')
                    td3_trainer.save_critic1_model(model_outdir, "td3_critic1_model_ep" + str(ep + 1) + '.pt')
                    td3_trainer.save_critic2_model(model_outdir, "td3_critic2_model_ep" + str(ep + 1) + '.pt')
                rospy.logwarn("DONE")
                data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1, ego_safety_score, social_safety_score, time_lapse]
                utils.record_data(data, result_outdir, "td3_training_test_real_world")
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                break

    env.reset()
