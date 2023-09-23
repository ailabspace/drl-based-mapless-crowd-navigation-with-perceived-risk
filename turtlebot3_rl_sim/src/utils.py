import os
import csv
import numpy as np
import math
import time
import copy
import pickle
import rospy
import colorsys
from math import pi
from collections import deque

from tf.transformations import euler_from_quaternion
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from visualization_msgs.msg import Marker


def load_q(file):
    """
        Loads trained Q-tables for Q-Learning or Sarsa algorithm.
    """
    q = None
    with open(file, 'rb') as f:
        q = pickle.loads(f.read())

    return q


def get_q(q, state, action):
    """
        Gets Q-values from the Q-table of Q-Learning or Sarsa algorithm.
    """
    return q.get((state, action), 0.0)


# Logging methods
def remove_logfile_if_exist(outdir, filename):
    try:
        os.remove(outdir + "/" + filename + ".csv")
    except OSError:
        pass


def remove_qfile_if_exist(outdir, file):
    try:
        os.remove(outdir + "/" + file + ".txt")
    except OSError:
        pass


def record_data(data, outdir, filename):
    file_exists = os.path.isfile(outdir + "/" + filename + ".csv")
    with open(outdir + "/" + filename + ".csv", "a") as fp:
        headers = ['episode_number', 'success_episode', 'failure_episode', 'episode_reward', 'episode_step',
                   'ego_safety_score', 'social_safety_score', 'timelapse']
        writer = csv.DictWriter(fp, delimiter=',', lineterminator='\n', fieldnames=headers)

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        wr = csv.writer(fp, dialect='excel')
        wr.writerow(data)


def get_sample_from_cluster(kmeans):  # numpy
    # Nice Pythonic way to get the indices of the points for each corresponding cluster
    mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

    # Transform this dictionary into list (if you need a list as result)
    dictlist = []
    for key, value in mydict.iteritems():
        value_list = value.tolist()
        temp = [key, value_list]
        dictlist.append(temp)

    return dictlist


def replace_obj_coord_with_centroid(centroid_coords, cluster_samples, obs_coords):
    centroid_sample_coord_list, centroid_sample_idx_list = [], []
    _obs_coords = obs_coords

    # Get cluster centroid coordinates and sample index
    for i in range(len(centroid_coords)):
        centroid_sample_coord_list.append(centroid_coords[i])
        centroid_sample_idx_list.append(cluster_samples[i][1])

    # Replace obstacle coordinate with centroid coordinates
    for i in range(len(centroid_sample_coord_list)):
        for j in range(len(centroid_sample_idx_list[i])):
            _obs_coords[centroid_sample_idx_list[i][j]] = centroid_sample_coord_list[i]

    return _obs_coords


def get_angle_from_point(current_orientation):
    current_ori_x = current_orientation.x
    current_ori_y = current_orientation.y
    current_ori_z = current_orientation.z
    current_ori_w = current_orientation.w

    orientation_list = [current_ori_x, current_ori_y, current_ori_z, current_ori_w]
    _, _, yaw = euler_from_quaternion(orientation_list)

    return yaw


def convert_laserscan_to_coordinate(scans, resolution, robot_pose, robot_yaw, max_angle, global_coord="gazebo"):
    # pos_y has -1.0 because the cartesian coordinate system in ros is negative in the x-axis right-hand side
    pos_list = []
    angle_increment = max_angle / (resolution - 1)
    if global_coord is not "gazebo":
        robot_pose_x = 0.0
        robot_pose_y = 0.0
    else:
        robot_pose_x = robot_pose.x
        robot_pose_y = robot_pose.y
    for i in range(resolution - 1):
        _angle = (i * angle_increment)
        pos_x = round(robot_pose_x + (scans[i] * math.cos(math.radians(_angle) - robot_yaw)), 3)
        pos_y = round(robot_pose_y + (scans[i] * math.sin(math.radians(_angle) - robot_yaw)) * -1.0, 3)
        pos_list.append([pos_x, pos_y])

    return pos_list


def get_heading_to_obs(current_position, current_orientation, obstacle_position):
    current_pos_x = current_position.x
    current_pos_y = current_position.y

    yaw = get_angle_from_point(current_orientation)
    goal_angle = math.atan2(obstacle_position[1] - current_pos_y, obstacle_position[0] - current_pos_x)

    heading = goal_angle - yaw
    if heading > pi:
        heading -= 2 * pi

    elif heading < -pi:
        heading += 2 * pi

    return heading


def get_obstacle_region(robot_pose, robot_heading, obs_pose, obs_scan, obs_heading):
    # Yaw is +180 to 0 and 0  to -180 (clockwise), we convert it to 0 to +360 (clockwise) degree format
    heading = convert_yaw_to_360deg(robot_heading)
    far_proximity = 0.6
    robot_radius = 0.16
    # far_proximity = 0.3
    # robot_radius = 0.12

    # Yaw is 0 to -180 (anti-clockwise) and  0 to +180 (clockwise).
    obs_heading = math.degrees(obs_heading)

    # Get the far end of the coordinate with proximity and heading information
    far_center_x = robot_pose.x - far_proximity * math.cos(math.radians(heading))
    far_center_y = robot_pose.y + far_proximity * math.sin(math.radians(heading))

    # Get the right and left coordinate of the far end coordinate in a circle boundary region
    far_right_x = far_center_x - robot_radius * math.cos(math.radians((90 + heading) % 360))
    far_right_y = far_center_y + robot_radius * math.sin(math.radians((90 + heading) % 360))
    far_left_x = far_center_x - robot_radius * math.cos(math.radians((270 + heading) % 360))
    far_left_y = far_center_y + robot_radius * math.sin(math.radians((270 + heading) % 360))

    # Get the right and left coordinate of the origin coordinate in a circle boundary region
    close_right_x = robot_pose.x - robot_radius * math.cos(math.radians((90 + heading) % 360))
    close_right_y = robot_pose.y + robot_radius * math.sin(math.radians((90 + heading) % 360))
    close_left_x = robot_pose.x - robot_radius * math.cos(math.radians((270 + heading) % 360))
    close_left_y = robot_pose.y + robot_radius * math.sin(math.radians((270 + heading) % 360))

    # Create a rectangular polygon area which is the front region of the robot and check if an obstacle is in it
    point = Point(obs_pose[0], obs_pose[1])
    front_right_polygon = Polygon(
        [(close_right_x, close_right_y), (far_right_x, far_right_y), (far_center_x, far_center_y),
         (robot_pose.x, robot_pose.y)])
    front_left_polygon = Polygon(
        [(robot_pose.x, robot_pose.y), (far_center_x, far_center_y), (far_left_x, far_left_y),
         (close_left_x, close_left_y)])
    right_polygon = Polygon(
        [(close_right_x, close_right_y), (close_right_x - (far_proximity - robot_radius), close_right_y),
         (close_right_x - (far_proximity - robot_radius), close_right_y + far_proximity),
         (close_right_x, close_right_y + far_proximity)])
    left_polygon = Polygon(
        [(close_left_x, close_left_y), (close_left_x - (far_proximity - robot_radius), close_left_y),
         (close_left_x - (far_proximity - robot_radius), close_left_y - far_proximity),
         (close_left_x, close_left_y - far_proximity)])

    # Conditions
    far_proximity_region = 0.3 < obs_scan < 0.6
    close_proximity_region = obs_scan < 0.3

    # Social region and proximity
    region = "OTHER"  # None
    if far_proximity_region:
        if front_right_polygon.contains(point):
            region = "FRF"
        if front_left_polygon.contains(point):
            region = "FLF"
        # if right_polygon.contains(point):
        #     region = "RF"
        # if left_polygon.contains(point):
        #     region = "LF"
    if close_proximity_region:
        if front_right_polygon.contains(point):
            region = "FRC"
        if front_left_polygon.contains(point):
            region = "FLC"
        # if right_polygon.contains(point):
        #     region = "RC"
        # if left_polygon.contains(point):
        #     region = "LC"

    return region


def get_obstacle_proximity(distance):
    if distance < 0.3:
        proximity = "Close"
    else:
        proximity = "Far"

    return proximity


def get_timestep_velocity(poses, timelapse):
    delta_pos_x = poses[1][0] - poses[0][0]
    delta_pos_y = poses[1][1] - poses[0][1]

    vel_x = delta_pos_x / timelapse
    vel_y = delta_pos_y / timelapse

    resultant_velocity = math.sqrt(math.pow(vel_x, 2) + math.pow(vel_y, 2))

    return resultant_velocity


def get_timestep_distance(poses):
    delta_pos_x = poses[1][0] - poses[0][0]
    delta_pos_y = poses[1][1] - poses[0][1]

    dist_x = delta_pos_x
    dist_y = delta_pos_y

    resultant_distance = math.sqrt(math.pow(dist_x, 2) + math.pow(dist_y, 2))

    return resultant_distance


def get_collision_point(agent_poses, obstacle_poses, obstacle_pose_radius):
    # Solve for x and y with straight line eqn of resultant velocity of agent with circle eqn of obstacle
    # based on collision cone algorithm

    # Circle eqn
    p = Point(obstacle_poses[0], obstacle_poses[1])
    circle = p.buffer(obstacle_pose_radius).boundary

    # Gradient is x1 - x2 / y1 - y2 because of gazebo's coordinate system
    try:
        gradient = (agent_poses[1][0] - agent_poses[0][0]) / agent_poses[1][1] - agent_poses[0][1]
    except ZeroDivisionError:
        gradient = 0.0
    # gradient = (agent_poses[1][0] - agent_poses[0][0]) / agent_poses[1][1] - agent_poses[0][1]
    constant_b = agent_poses[0][0] - (gradient * (agent_poses[0][1]))

    # Solve for y with x starting from agent's x pose and up to the max laser scan range
    # Why not take +ve x values? Because there are no reverse motion here for the agent and the gazebo coordinate
    # system is rotated (x and y-axis)
    # -1 is the increment of the loop
    for agent_pose_x2 in range(int(math.ceil(agent_poses[0][0] + 3.5)), int(math.floor(agent_poses[0][0] - 3.5)), -1):
        agent_pose_y2 = (agent_pose_x2 * gradient) + constant_b

        # Line eqn
        line = LineString([(agent_poses[0][0], agent_poses[0][1]), (agent_pose_x2, agent_pose_y2)])

        i = circle.intersection(line)

        if str(i) != 'LINESTRING EMPTY':
            try:
                _p1 = list(i.geoms[0].coords[0])
                _p2 = list(i.geoms[1].coords[0])
                _dist_robot_p1 = math.hypot(agent_poses[0][0] - _p1[0], agent_poses[0][1] - _p1[1])
                _dist_robot_p2 = math.hypot(agent_poses[0][0] - _p2[0], agent_poses[0][1] - _p2[1])
                dist_to_cp = min(_dist_robot_p1, _dist_robot_p2)
                break
            except Exception:
                dist_to_cp = None
                break
        else:
            dist_to_cp = None

    return dist_to_cp


def get_local_goal_waypoints(agent_pose, goal_pose, boundary_radius, epsilon=0.0):
    # Implementation is based on getting the point of intersection between a line from the agent's position
    # to the goal position where the circle region is from the agent position with a radius according to the max
    # laser scan distance
    # If no intersection is found, then use the original goal point as the next waypoint
    p = Point(agent_pose[0], agent_pose[1])
    c = p.buffer(boundary_radius).boundary
    l = LineString([(agent_pose[0], agent_pose[1]), (goal_pose[0], goal_pose[1])])
    i = c.intersection(l)

    if str(i) != 'LINESTRING EMPTY':
        try:
            goal_waypoints = [i.x, i.y]
        except Exception:
            goal_waypoints = [-(goal_pose[0] + epsilon), goal_pose[1] + epsilon]
    else:
        goal_waypoints = [-(goal_pose[0] + epsilon), goal_pose[1] + epsilon]

    return goal_waypoints


def compute_collision_prob(time_to_collision):
    if time_to_collision is not None:
        collision_probability = min(1, 0.15 / time_to_collision)
    else:
        collision_probability = 0.0

    return collision_probability


def compute_general_collision_prob(scans, max_range, min_range):
    """

    Args:
        scans: the laser scans of the agent
        max_range: the distance considered to have 0% collision probability
        min_range: the distance considered to have 100% collision probability

    Returns: general collision probability (0 - 1.0)

    """
    # minimum_scan = min(scans)
    minimum_scan = scans

    if minimum_scan > max_range:
        general_collision_probability = 0.0
    else:
        general_collision_probability = (max_range - minimum_scan) / (max_range - min_range)

    return general_collision_probability


def init_deque_list(size):
    _list = []
    for i in range(size):
        _list.append(deque([]))

    return _list


def convert_yaw_to_360deg(yaw):
    _yaw = math.degrees(yaw)
    _yaw = yaw * 180.0 / math.pi
    _yaw -= 180.0
    _yaw = abs(_yaw)
    if _yaw < 0:
        _yaw += 360.0

    return _yaw


def append_to_dynamic_list(list_obj, list_idx, item):
    try:
        list_obj[list_idx].append(item)
    except:
        list_obj.append([])
        list_obj[list_idx].append(item)


def get_scan_ranges(scan, scan_ranges, max_range):
    _scan_range, scan_range = [], []
    for i in range(scan_ranges):
        if scan.ranges[i] == float('Inf') or scan.ranges[i] == float('inf'):
            _scan_range.append(max_range)
        elif np.isnan(scan.ranges[i]) or scan.ranges[i] == float('nan'):
            _scan_range.append(0)
        elif scan.ranges[i] == 0.0:  # for real world with limited fov sensor
            _scan_range.append(max_range)
        elif scan.ranges[i] > max_range:  # for real world with limited fov sensor
            _scan_range.append(max_range)
        else:
            _scan_range.append(scan.ranges[i])

    _scan_range.reverse()
    scan_range = _scan_range[:-1]

    return scan_range


def estimate_num_obs_scans(dist_to_object, max_range, min_range):
    """
        For obstacle type object:
            At max range (0.6m) - 32 scans
            At min range (0.136m) - 3 scans
    """
    num_obs_scan = 3 + math.floor(29 * (max_range - dist_to_object) / (max_range - min_range))
    return num_obs_scan


def compute_average_bounding_box_size(ground_truth_poses):
    dist_between_scans = []
    for i in range(len(ground_truth_poses)):
        if i == len(ground_truth_poses) - 1:
            _dist_between_scans = math.hypot(ground_truth_poses[i][0] - ground_truth_poses[0][0],
                                             ground_truth_poses[i][1] - ground_truth_poses[0][1])
            dist_between_scans.append(_dist_between_scans)
        else:
            _dist_between_scans = math.hypot(ground_truth_poses[i][0] - ground_truth_poses[i + 1][0],
                                             ground_truth_poses[i][1] - ground_truth_poses[i + 1][1])
            dist_between_scans.append(_dist_between_scans)

    average_dist = float(sum(dist_between_scans)) / len(dist_between_scans)

    return average_dist


def _get_bounding_box(scan, bounding_box_size):
    x_pos_plus = scan[0] + bounding_box_size
    x_pos_minus = scan[0] - bounding_box_size
    y_pos_plus = scan[1] + bounding_box_size
    y_pos_minus = scan[1] - bounding_box_size

    bounding_box = [[x_pos_plus, y_pos_plus], [x_pos_plus, y_pos_minus], [x_pos_minus, y_pos_minus],
                    [x_pos_minus, y_pos_plus]]

    return bounding_box


# Hungarian algorithm association (IOU) score
def is_associated(scan1, scan2, bounding_box_size):
    # Get bounding box
    box_1 = _get_bounding_box(scan1, bounding_box_size * 1.0)
    box_2 = _get_bounding_box(scan2, bounding_box_size * 1.0)

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    iou = round(poly_1.intersection(poly_2).area / poly_1.union(poly_2).area, 3)
    # print(iou)
    if iou > 0.0:
        return True
    else:
        return False


def get_iou(scan1, scan2, bounding_box_size):
    box_1 = _get_bounding_box(scan1, bounding_box_size * 1.0)
    box_2 = _get_bounding_box(scan2, bounding_box_size * 1.0)

    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)

    iou = round(poly_1.intersection(poly_2).area / poly_1.union(poly_2).area, 3)

    return iou


def check_list(sample_list, index_to_check):
    try:
        new_list = copy.deepcopy(sample_list)
        val = new_list[index_to_check].append('random')
        return True
    except:
        return False


def create_rviz_visualization_text_marker(marker, robot_pose, obs_pose, cp, mtype="obstacle"):
    # Pose accepts [x, y] and cp accepts floats
    # CP float number is normalized to a range of RGB, where
    # red is the highest CP while green is the least
    marker.header.frame_id = "/base_footprint"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    if mtype is "obstacle":
        marker.type = 9
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.101
        marker.scale.y = 0.101
        marker.scale.z = 0.15

        # Set the color, from HSL to RGB
        if cp is None:
            cp = 0.0
        # print("CP: ", cp)
        h_value = 100 - (cp * 100)
        s_value = 1.0
        l_value = 1.0
        rgb_values = colorsys.hsv_to_rgb(h_value, s_value, l_value)
        # print(rgb_values)
        marker.color.r = 1.0  # rgb_values[0]
        marker.color.g = 1.0  # rgb_values[1]
        marker.color.b = 1.0  # 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = -obs_pose[0] + robot_pose[0] + 0.1
        marker.pose.position.y = -obs_pose[1] + robot_pose[1]
        marker.pose.position.z = 0.3
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # marker text
        marker.text = str(round(cp * 100, 1)) + "%"

    return marker


def create_rviz_visualization_shape_marker(marker, robot_pose, obs_pose, cp, mtype="obstacle", goal_pose=None):
    # Pose accepts [x, y] and cp accepts floats
    # CP float number is normalized to a range of RGB, where
    # red is the highest CP while green is the least
    if goal_pose is None:
        goal_pose = [2, 2]
    marker.header.frame_id = "/base_footprint"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    if mtype is "obstacle":
        marker.type = 3
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.055
        marker.scale.y = 0.055
        marker.scale.z = 0.2

        # Set the color, from HSL to RGB
        if cp is None:
            cp = 0.0
        # print("CP: ", cp)
        h_value = 100 - (cp * 100)
        s_value = 1.0
        l_value = 1.0
        rgb_values = colorsys.hsv_to_rgb(h_value, s_value, l_value)
        # print(rgb_values)

        marker.color.r = 0.0  # rgb_values[0]
        marker.color.g = 1.0  # rgb_values[1]
        marker.color.b = 0.0  # 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = -obs_pose[0] + robot_pose[0]
        marker.pose.position.y = -obs_pose[1] + robot_pose[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

    else:
        if goal_pose == [2, 2]:
            marker.type = 1
        else:
            marker.type = 2
        marker.id = 0

        # Set the scale of the marker
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.05

        # Set the color, from HSL to RGB
        if goal_pose == [2, 2]:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = goal_pose[0] + robot_pose[0]
        marker.pose.position.y = -(goal_pose[1]) + robot_pose[1]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

    return marker