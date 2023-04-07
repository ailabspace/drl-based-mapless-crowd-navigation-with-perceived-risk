#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
import time
from math import pi
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped

from collections import deque
import utils
import rospkg


class Env:
    def __init__(self, action_dim=2, max_step=200):
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.action_dim = action_dim
        # Keys CTRL + c will stop script
        rospy.on_shutdown(self.shutdown)

        # Added
        self.orientation = 0.0
        self.previous_heading = 0.0
        self.previous_distance = 0.0
        self.episode_success = False
        self.episode_failure = False
        self.desired_point = Point()
        self.desired_point.x = rospy.get_param("/turtlebot3/desired_pose/x")
        self.desired_point.y = rospy.get_param("/turtlebot3/desired_pose/y")
        self.desired_point.z = rospy.get_param("/turtlebot3/desired_pose/z")
        self.linear_forward_speed = rospy.get_param('/turtlebot3/linear_forward_speed')
        self.linear_turn_speed = rospy.get_param('/turtlebot3/linear_turn_speed')
        self.angular_speed = rospy.get_param('/turtlebot3/angular_speed')
        self.scan_ranges = rospy.get_param('/turtlebot3/scan_ranges')
        self.max_steps = max_step
        self.done = False
        self.robot_odometry = None

        # Object detection and tracking
        # rospy.Subscriber('gazebo/model_states', ModelStates, self.obstacle_pose_callback)

        self.pub_obs1_pose = rospy.Publisher('/obstacle_poses/1', PointStamped, queue_size=1)
        self.pub_obs2_pose = rospy.Publisher('/obstacle_poses/2', PointStamped, queue_size=1)
        self.pub_obs3_pose = rospy.Publisher('/obstacle_poses/3', PointStamped, queue_size=1)
        self.pub_obs4_pose = rospy.Publisher('/obstacle_poses/4', PointStamped, queue_size=1)
        self.pub_obs5_pose = rospy.Publisher('/obstacle_poses/5', PointStamped, queue_size=1)
        self.pub_obs6_pose = rospy.Publisher('/obstacle_poses/6', PointStamped, queue_size=1)
        self.pub_obs7_pose = rospy.Publisher('/obstacle_poses/7', PointStamped, queue_size=1)

        self.pub_cluster_obs1_pose = rospy.Publisher('/cluster_obstacle_poses/1', PointStamped, queue_size=1)
        self.pub_cluster_obs2_pose = rospy.Publisher('/cluster_obstacle_poses/2', PointStamped, queue_size=1)
        self.pub_cluster_obs3_pose = rospy.Publisher('/cluster_obstacle_poses/3', PointStamped, queue_size=1)
        self.pub_cluster_obs4_pose = rospy.Publisher('/cluster_obstacle_poses/4', PointStamped, queue_size=1)
        self.pub_cluster_obs5_pose = rospy.Publisher('/cluster_obstacle_poses/5', PointStamped, queue_size=1)
        self.pub_cluster_obs6_pose = rospy.Publisher('/cluster_obstacle_poses/6', PointStamped, queue_size=1)
        self.pub_cluster_obs7_pose = rospy.Publisher('/cluster_obstacle_poses/7', PointStamped, queue_size=1)

        # object clustering
        self.obj_clusters = 7

        # Reward shaping based on moving obstacle region and proximity
        self.collision_prob = None
        self.goal_reaching_prob = None
        self.general_collision_prob = None
        self.positive_agent_obs_dist_change = None
        self.closest_obstacle_dist_change = None
        self.closest_obstacle_region = None
        self.closest_obstacle_proximity = None
        self.closest_obstacle_pose = None

        # Deque lists to compare items between time steps
        self.agent_pose_deque = deque([])
        self.obstacle_pose_deque = utils.init_deque_list(self.scan_ranges - 1)
        self.obstacle_scan_range_deque = [deque([]), deque([]), deque([]), deque([]), deque([]), deque([]), deque([])]
        self.agent_obstacle_dist_deque = [deque([]), deque([]), deque([]), deque([]), deque([]), deque([]), deque([])]
        self.vel_t0 = -1  # Store starting time when vel cmd is executed, to get a time step length
        self.goal_vel_t0 = -1

        # Temporary (delete)
        self.step_reward_count = 0
        self.dtg_reward_count = 0
        self.htg_reward_count = 0
        self.dtg_penalty_count = 0
        self.htg_penalty_count = 0
        self.forward_action_reward_count = 0
        self.left_turn_action_reward_count = 0
        self.right_turn_action_reward_count = 0
        self.weak_left_turn_action_reward_count = 0
        self.weak_right_turn_action_reward_count = 0
        self.strong_left_turn_action_reward_count = 0
        self.strong_right_turn_action_reward_count = 0
        self.rotate_in_place_action_reward_count = 0
        self.social_nav_reward_count = 0

        self.last_action = "FORWARD"

        # Set the logging system
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('turtlebot3_rl_sim')
        self.result_outdir = pkg_path + '/src/results/td3_final' + '/'

    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot")
        self.pub_cmd_vel.publish(Twist())
        rospy.sleep(1)

    def obstacle_pose_callback(self, msg):
        obs_to_robot_dist = []

        if msg is not None:
            obs_1_index = msg.name.index('obstacle_1')
            obs_2_index = msg.name.index('obstacle_2')
            obs_3_index = msg.name.index('obstacle_3')
            obs_4_index = msg.name.index('obstacle_4')
            obs_5_index = msg.name.index('obstacle_5')
            obs_6_index = msg.name.index('obstacle_6')
            obs_7_index = msg.name.index('obstacle_7')

            self.obs_1_pose = msg.pose[obs_1_index]
            self.obs_2_pose = msg.pose[obs_2_index]
            self.obs_3_pose = msg.pose[obs_3_index]
            self.obs_4_pose = msg.pose[obs_4_index]
            self.obs_5_pose = msg.pose[obs_5_index]
            self.obs_6_pose = msg.pose[obs_6_index]
            self.obs_7_pose = msg.pose[obs_7_index]

            self.obs_1_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_1_pose.position.x,
                                                                self.obs_1_pose.position.y)
            self.obs_2_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_2_pose.position.x,
                                                                self.obs_3_pose.position.y)
            self.obs_3_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_3_pose.position.x,
                                                                self.obs_3_pose.position.y)
            self.obs_4_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_4_pose.position.x,
                                                                self.obs_4_pose.position.y)
            self.obs_5_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_5_pose.position.x,
                                                                self.obs_5_pose.position.y)
            self.obs_6_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_6_pose.position.x,
                                                                self.obs_6_pose.position.y)
            self.obs_7_to_robot_xy = self.get_robot_obs_xy_diff(self.position.x, self.position.y,
                                                                self.obs_7_pose.position.x,
                                                                self.obs_7_pose.position.y)

            self.obs_1_to_robot_dist = math.hypot(self.obs_1_to_robot_xy[0], self.obs_1_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_1', "moving", [self.obs_1_pose.position.x, self.obs_1_pose.position.y],
                                      self.obs_1_to_robot_dist])
            self.obs_2_to_robot_dist = math.hypot(self.obs_2_to_robot_xy[0], self.obs_2_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_2', "moving", [self.obs_2_pose.position.x, self.obs_2_pose.position.y],
                                      self.obs_2_to_robot_dist])
            self.obs_3_to_robot_dist = math.hypot(self.obs_3_to_robot_xy[0], self.obs_3_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_3', "moving", [self.obs_3_pose.position.x, self.obs_3_pose.position.y],
                                      self.obs_3_to_robot_dist])
            self.obs_4_to_robot_dist = math.hypot(self.obs_4_to_robot_xy[0], self.obs_4_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_4', "moving", [self.obs_4_pose.position.x, self.obs_4_pose.position.y],
                                      self.obs_6_to_robot_dist])
            self.obs_7_to_robot_dist = math.hypot(self.obs_7_to_robot_xy[0], self.obs_7_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_7', "moving", [self.obs_7_pose.position.x, self.obs_7_pose.position.y],
                                      self.obs_7_to_robot_dist])
            self.obs_5_to_robot_dist = math.hypot(self.obs_5_to_robot_xy[0], self.obs_5_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_5', "moving", [self.obs_5_pose.position.x, self.obs_5_pose.position.y],
                                      self.obs_5_to_robot_dist])
            self.obs_6_to_robot_dist = math.hypot(self.obs_6_to_robot_xy[0], self.obs_6_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_6', "moving", [self.obs_6_pose.position.x, self.obs_6_pose.position.y],
                                      self.obs_6_to_robot_dist])
            self.obs_7_to_robot_dist = math.hypot(self.obs_7_to_robot_xy[0], self.obs_7_to_robot_xy[1])
            obs_to_robot_dist.append(['obstacle_7', "moving", [self.obs_7_pose.position.x, self.obs_7_pose.position.y],
                                      self.obs_7_to_robot_dist])

            self.obs_to_robot_dist = obs_to_robot_dist

    def get_robot_obs_xy_diff(self, robot_pose_x, robot_pose_y, obs_pose_x, obs_pose_y):
        """
        Args:
            robot_pose_x: robot's x position
            robot_pose_y: robot's y position
            obs_pose_x: obstacle's x position
            obs_pose_y: obstacle's y position

        Returns: returns distance in x and y axis between robot and obstacle

        """
        robot_obs_x = abs(robot_pose_x - obs_pose_x)
        robot_obs_y = abs(robot_pose_y - obs_pose_y)

        return [robot_obs_x, robot_obs_y]

    def get_distance_from_point(self, pstart, p_end):
        a = np.array((pstart.x, pstart.y, pstart.z))
        b = np.array((p_end.x, p_end.y, p_end.z))

        distance = np.linalg.norm(a - b)

        return distance

    def get_distance_to_goal(self, current_position):
        distance = self.get_distance_from_point(current_position,
                                                self.desired_point)

        return distance

    def get_angle_from_point(self, current_orientation):
        current_ori_x = current_orientation.x
        current_ori_y = current_orientation.y
        current_ori_z = current_orientation.z
        current_ori_w = current_orientation.w

        orientation_list = [current_ori_x, current_ori_y, current_ori_z, current_ori_w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        return yaw

    def get_heading_to_goal(self, current_position, current_orientation):
        current_pos_x = current_position.x
        current_pos_y = current_position.y

        yaw = self.get_angle_from_point(current_orientation)
        goal_angle = math.atan2(self.desired_point.y - current_pos_y, self.desired_point.x - current_pos_x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        return heading

    def get_robot_yaw(self, odom_orientation):
        yaw = self.get_angle_from_point(odom_orientation)

        return yaw

    def get_robot_odom(self, odom_pose, odom_orientation):
        _robot_pose_x = round(odom_pose.x, 3)
        _robot_pose_y = round(odom_pose.y, 3)
        _robot_yaw = self.get_robot_yaw(odom_orientation)

        self.robot_odometry = [round(odom_pose.x, 3), round(odom_pose.y, 3), _robot_yaw]

    def get_odometry(self, odom):
        self.position = odom.pose.pose.position
        self.orientation = odom.pose.pose.orientation

    def get_state(self, scan, step_counter=0, action=[0, 0]):
        _scan_range = []
        distance_to_goal = round(self.get_distance_to_goal(self.position), 2)
        heading_to_goal = round(self.get_heading_to_goal(self.position, self.orientation), 2)
        min_range = 0.105  # 0.125 # 0.136

        self.get_robot_odom(self.position, self.orientation)
        data = [step_counter, self.robot_odometry[0], self.robot_odometry[1], math.degrees(self.robot_odometry[2])]
        utils.record_data(data, self.result_outdir, "td3_training_test_trajectory")

        # Get scan ranges (to know when to stop an episode when the agent is too close to an obstacle)
        for i in range(self.scan_ranges):
            if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
                _scan_range.append(0.6)
            elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
                _scan_range.append(0)
            else:
                _scan_range.append(scan.ranges[i])

        # Change the scan ranges to reverse the order and remove the final scan
        # The scan reads in an anti-clockwise manner and the final scan is the same as the first scan, so it is omitted.
        _scan_range.reverse()
        scan_range = _scan_range[:-1]

        if not self.done:
            if min_range > min(scan_range) > 0:
                print("DONE: MINIMUM RANGE")
                self.done = True

            if self.is_in_desired_position(self.position):
                print("DONE: IN DESIRED POSITION")
                self.done = True

            if step_counter >= self.max_steps:
                print("DONE: STEP COUNTER > MAX STEP")
                self.done = True

        agent_position = [round(self.position.x, 3), round(self.position.y, 3)]
        goal_heading_distance = [heading_to_goal, distance_to_goal]
        general_obs_distance = [round(num, 3) for num in scan_range]

        # state = general_obs_distance + goal_heading_distance
        state = general_obs_distance + goal_heading_distance + agent_position

        return state, self.done

    def compute_reward(self, state, done):
        current_distance = state[-1]
        current_heading = state[-2]

        distance_difference = current_distance - self.previous_distance
        heading_difference = current_heading - self.previous_heading

        # ADDED: step penalty
        action_reward = 0
        htg_reward = 0
        dtg_reward = 0
        step_reward = 0  # -2

        # Action reward
        if self.last_action == "FORWARD":
            self.forward_action_reward_count += 1
            action_reward = 1
        if self.last_action == "TURN_LEFT":
            self.left_turn_action_reward_count += 1
            action_reward = 1
        if self.last_action == "TURN_RIGHT":
            self.right_turn_action_reward_count += 1
            action_reward = 1

        # Distance to goal reward
        if distance_difference > 0:
            self.dtg_penalty_count += 1
            dtg_reward = 0
        if distance_difference < 0:
            self.dtg_reward_count += 1
            dtg_reward = 1

        # Heading to goal reward
        if heading_difference > 0:
            if current_heading > 0 and self.previous_heading < 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading < 0 and self.previous_heading < 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading < 0 and self.previous_heading > 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading > 0 and self.previous_heading > 0:
                self.htg_penalty_count += 1
                htg_reward = 0
        if heading_difference < 0:
            if current_heading < 0 and self.previous_heading > 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading > 0 and self.previous_heading > 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading > 0 and self.previous_heading < 0:
                self.htg_reward_count += 1
                htg_reward = 1
            if current_heading < 0 and self.previous_heading < 0:
                self.htg_penalty_count += 1
                htg_reward = 0

        robot_pose = [round(self.position.x, 3), round(self.position.y, 3)]

        non_terminating_reward = dtg_reward + htg_reward + step_reward
        self.step_reward_count += 1

        if self.last_action is not None:
            reward = non_terminating_reward

        self.previous_distance = current_distance
        self.previous_heading = current_heading

        if done:
            if self.is_in_desired_position(self.position):
                rospy.loginfo("Reached goal position!!")
                self.episode_failure = False
                self.episode_success = True
                goal_reward = 200
                reward = goal_reward + non_terminating_reward
            else:
                rospy.loginfo("Collision!!")
                self.episode_failure = True
                self.episode_success = False
                collision_reward = -200
                reward = collision_reward + non_terminating_reward
            self.pub_cmd_vel.publish(Twist())

        return reward, done

    def step(self, action, step_counter, mode="discrete"):
        if mode is "discrete":
            if action == 0:  # FORWARD
                linear_speed = 0.22  # self.linear_forward_speed
                angular_speed = 0.0
                self.last_action = "FORWARDS"
            elif action == 1:  # LEFT
                linear_speed = 0.22  # self.linear_turn_speed
                angular_speed = 2.0  # self.angular_speed
                self.last_action = "TURN_LEFT"
            elif action == 2:  # RIGHT
                linear_speed = 0.22  # self.linear_turn_speed
                angular_speed = -1 * 2.0  # self.angular_speed
                self.last_action = "TURN_RIGHT"
        else:
            linear_speed = action[0]
            angular_speed = action[1]
            if linear_speed >= 0 and (((1.0 / 32.0) * -2.0) <= angular_speed <= (1.0 / 32.0) * 2.0):
                # if linear_speed >= 0 and angular_speed == 0:
                self.last_action = "FORWARD"
            elif linear_speed >= 0 and angular_speed > 0:
                self.last_action = "TURN_LEFT"
            elif linear_speed >= 0 and angular_speed < 0:
                self.last_action = "TURN_RIGHT"

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_speed
        vel_cmd.angular.z = angular_speed

        if self.vel_t0 is -1:  # Reset when deque is full
            self.vel_t0 = time.time()  # Start timer to get the timelapse between two positions of agent
        self.pub_cmd_vel.publish(vel_cmd)
        time.sleep(0.15)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.get_state(data, step_counter, action)
        reward, done = self.compute_reward(state, done)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            print("RESET PROXY")
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        # Get initial heading and distance to goal
        self.previous_distance = self.get_distance_to_goal(self.position)
        self.previous_heading = self.get_heading_to_goal(self.position, self.orientation)
        state, _ = self.get_state(data)

        # Temporary (delete)
        self.step_reward_count = 0
        self.dtg_reward_count = 0
        self.htg_reward_count = 0
        self.dtg_penalty_count = 0
        self.htg_penalty_count = 0
        self.forward_action_reward_count = 0
        self.strong_right_turn_action_reward_count = 0
        self.strong_left_turn_action_reward_count = 0
        self.weak_right_turn_action_reward_count = 0
        self.weak_left_turn_action_reward_count = 0
        self.rotate_in_place_action_reward_count = 0
        return np.asarray(state)

    def get_episode_status(self):

        return self.episode_success, self.episode_failure

    def get_odometry_data(self):

        return self.robot_odometry

    def is_in_desired_position(self, current_position, epsilon=0.20):  # originally 0.05, changed to 0.20
        is_in_desired_pos = False

        x_pos_plus = self.desired_point.x + epsilon
        x_pos_minus = self.desired_point.x - epsilon
        y_pos_plus = self.desired_point.y + epsilon
        y_pos_minus = self.desired_point.y - epsilon

        x_current = current_position.x
        y_current = current_position.y

        x_pos_are_close = (x_current <= x_pos_plus) and (x_current > x_pos_minus)
        y_pos_are_close = (y_current <= y_pos_plus) and (y_current > y_pos_minus)

        is_in_desired_pos = x_pos_are_close and y_pos_are_close

        return is_in_desired_pos
