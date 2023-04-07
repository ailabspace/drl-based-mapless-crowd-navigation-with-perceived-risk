#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import json
import deepq
import time
import rospy
import numpy
import rospkg
import utils
from environment_stage_1_original import Env

if __name__ == '__main__':
    rospy.init_node('dqn_training', anonymous=True, log_level=rospy.WARN)

    # Init environment
    max_step = rospy.get_param("/turtlebot3/nsteps")
    env = Env(action_dim=3, max_step=max_step)
    stage_name = rospy.get_param("/turtlebot3/stage_name")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('turtlebot3_rl_sim')
    result_outdir = pkg_path + '/src/results/dqn' + '/'
    model_outdir = pkg_path + '/src/models/dqn' + '/'
    model_param_path = pkg_path + '/src/models/dqn'+ '/' + '/dqn_model_ep'

    # Remove log file if exist
    # utils.remove_logfile_if_exist(result_outdir, "dqn_training")

    continue_execution = False
    resume_epoch = 1500
    resume_path = model_param_path + str(resume_epoch)
    weights_path = resume_path + '.h5'
    params_json = resume_path + '.json'

    if not continue_execution:
        # Each time we take a sample and update our weights it is called a mini-batch.
        # Each time we run through the entire dataset, it's called an epoch.
        # PARAMETER LIST
        nepisodes = rospy.get_param("/turtlebot3/nepisodes")
        nsteps = rospy.get_param("/turtlebot3/nsteps")
        explorationRate = rospy.get_param("/turtlebot3/epsilon")
        learningRate = rospy.get_param("/turtlebot3/alpha")
        discountFactor = rospy.get_param("/turtlebot3/gamma")
        epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")
        updateTargetNetwork = 10000
        minibatch_size = 64
        learnStart = 64
        memorySize = 1000000
        network_inputs = 361 #54
        network_outputs = 3
        network_structure = [300, 300]
        current_epoch = 0

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        with open(params_json) as outfile:
            d = json.load(outfile)
            nepisodes = d.get('nepisodes')
            nsteps = d.get('nsteps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')
            epsilon_discount = rospy.get_param("/turtlebot3/epsilon_discount")

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure, training=True)
        deepQ.loadWeights(weights_path)

    stepCounter = 0

    for ep in range(nepisodes):
        cumulated_reward = 0

        if explorationRate > 0.05:
            explorationRate *= epsilon_discount

        # Initialize the environment and get first state of the robot
        observation = env.reset()
        time.sleep(0.1)  # Give time for RL to reset the agent's position
        env.done = False
        _observation = numpy.array(observation)
        state = _observation

        for step in range(nsteps):
            rospy.logwarn("EPISODE: " + str(ep + 1) + " | STEP: " + str(step + 1))

            state = numpy.array(state)
            qValues = deepQ.getQValues(state)
            action = deepQ.selectAction(qValues, explorationRate)
            observation, reward, done = env.step(action, step + 1)
            success_episode, failure_episode = env.get_episode_status()
            robot_odom = env.get_odometry_data()

            cumulated_reward += reward

            next_state = observation
            deepQ.addMemory(state, action, reward, next_state, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else:
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            if not done:
                rospy.logwarn("NOT DONE")
                stepCounter += 1
                if stepCounter % updateTargetNetwork == 0:
                    deepQ.updateTargetNetwork()
                state = next_state

            if done:
                # Debugging purposes
                if (step + 1) <= 2:
                    env.shutdown()
                    raw_input("Press Enter to continue...")
                if (ep + 1) % 100 == 0:
                    # save model weights and monitoring data every 100 epochs.
                    deepQ.saveModel(model_outdir, "dqn_model_ep" + str(ep + 1) + '.h5')
                    # save simulation parameters.
                    parameter_keys = ['nepisodes', 'nsteps', 'updateTargetNetwork', 'explorationRate', 'minibatch_size',
                                      'learnStart', 'learningRate', 'discountFactor', 'memorySize',
                                      'network_inputs', 'network_outputs', 'network_structure', 'current_epoch']
                    parameter_values = [nepisodes, nsteps, updateTargetNetwork, explorationRate, minibatch_size,
                                        learnStart, learningRate, discountFactor, memorySize, network_inputs,
                                        network_outputs, network_structure, ep + 1]
                    parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                    with open(model_param_path + str(ep + 1) + '.json', 'w') as outfile:
                        json.dump(parameter_dictionary, outfile)
                rospy.logwarn("DONE")
                data = [ep + 1, success_episode, failure_episode, cumulated_reward, step + 1]
                utils.record_data(data, result_outdir, "dqn_training_test_trajectory")
                print("EPISODE REWARD: ", cumulated_reward)
                print("EPISODE STEP: ", step + 1)
                print("EPISODE SUCCESS: ", success_episode)
                print("EPISODE FAILURE: ", failure_episode)
                break

    env.reset()
