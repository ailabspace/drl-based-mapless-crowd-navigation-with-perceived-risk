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
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

import random


class Moving():
    def __init__(self):
        # ROS node initialization
        rospy.init_node('simulate_crowd', disable_signals=True)
        self.node_name = rospy.get_name()
        rospy.logwarn("%s node started" % self.node_name)

        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        rospy.Subscriber('gazebo/model_states', ModelStates, self.model_states_callback)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.get_odometry)

        self.position = Pose()
        self.orientation = 0.0

        self.model = rospy.wait_for_message('gazebo/model_states', ModelStates)

    def spin(self):
        try:
            r = rospy.Rate(10)

            while not rospy.is_shutdown():
                try:
                    self.main()
                    r.sleep()
                except KeyboardInterrupt:
                    break
        except rospy.ROSInterruptException:
            pass

    def model_states_callback(self, msg):
        if msg is not None:
            model_1_index = msg.name.index('obstacle_1')
            model_3_index = msg.name.index('obstacle_3')
            model_4_index = msg.name.index('obstacle_4')
            model_5_index = msg.name.index('obstacle_5')
            model_7_index = msg.name.index('obstacle_7')
            model_9_index = msg.name.index('obstacle_9')
            model_11_index = msg.name.index('obstacle_11')
            model_12_index = msg.name.index('obstacle_12')

            self.model_1_pose = msg.pose[model_1_index]
            self.model_3_pose = msg.pose[model_3_index]
            self.model_4_pose = msg.pose[model_4_index]
            self.model_5_pose = msg.pose[model_5_index]
            self.model_7_pose = msg.pose[model_7_index]
            self.model_9_pose = msg.pose[model_9_index]
            self.model_11_pose = msg.pose[model_11_index]
            self.model_12_pose = msg.pose[model_12_index]

    def get_odometry(self, odom):
        self.position = odom.pose.pose.position
        self.orientation = odom.pose.pose.orientation

    def moving_1(self):
        start_time = time.time()
        # max vel = square root of vel x^2 and vel y^2
        random_vel_x = [random.uniform(-0.2, 0.2) for i in range(14)]  # Default: +/- 0.254
        random_vel_y = [random.uniform(-0.2, 0.2) for i in range(14)]  # Default: +/- 0.254

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            # print(elapsed_time)
            if elapsed_time > 0.5:  # Default: 2.25, 0.5
                break
            else:
                self.move_model('obstacle_1', self.model_1_pose, 0.0, 0.1, 0)
                self.move_model('obstacle_3', self.model_3_pose, -0.1, 0.1, 0)
                self.move_model('obstacle_4', self.model_4_pose, 0.0, 0.1, 0)
                self.move_model('obstacle_5', self.model_5_pose, -0.1, 0.0, 0)
                self.move_model('obstacle_7', self.model_7_pose, 0.0, -0.1, 0)
                self.move_model('obstacle_9', self.model_9_pose, 0.0, -0.1, 0)
                self.move_model('obstacle_11', self.model_11_pose, -0.1, -0.1, 0)
                self.move_model('obstacle_12', self.model_12_pose, -0.1, -0.1, 0)


    def move_model(self, model_name, pose, vel_x, vel_y, vel_z):
        obstacle = ModelState()
        obstacle.model_name = model_name
        obstacle.pose = pose
        obstacle.twist = Twist()
        # if abs(pose.position.x - self.position.x) < 0.3 and abs(pose.position.y - self.position.y) < 0.3:
        #     # print("OBSTACLE TOO CLOSE SO STOPPING")
        #     # vel_x = -vel_x
        #     # vel_y = -vel_y
        #     vel_x = 0
        #     vel_y = 0
        #     # vel_z = 0
        obstacle.twist.linear.x = vel_x
        obstacle.twist.linear.y = vel_y
        obstacle.twist.angular.z = vel_z
        self.pub_model.publish(obstacle)
        time.sleep(0.1)

    def main(self):
        self.moving_1()


if __name__ == '__main__':
    try:
        moving = Moving()
        time.sleep(1)
        moving.spin()

    except rospy.ROSInterruptException:
        pass
