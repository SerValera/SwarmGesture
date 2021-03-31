#!/usr/bin/env python
# license removed for brevity

from nav_msgs.msg import Path
# make key points

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import csv
import math
import time
from collections import deque

import re

# --real time plot lib---
# https://www.galaxysofts.com/new/python-creating-a-real-time-3d-plot/
from itertools import count

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


rospy.init_node('publish_position', anonymous=True)
pub1 = rospy.Publisher('/drone1', PoseStamped, queue_size=10)
pub2 = rospy.Publisher('/drone2', PoseStamped, queue_size=10)
pub3 = rospy.Publisher('/drone3', PoseStamped, queue_size=10)
pub4 = rospy.Publisher('/drone4', PoseStamped, queue_size=10)
pub5 = rospy.Publisher('/drone5', PoseStamped, queue_size=10)

pub_catch_pos = rospy.Publisher('/catch_pos', PoseStamped, queue_size=10)

pub_for_drawing = rospy.Publisher('/drone_drawing', String, queue_size=10)

pub_add_hand = rospy.Publisher('add_hand', String, queue_size=10)

pub_gesture = rospy.Publisher('/gesture_number', String, queue_size=10)

rate = rospy.Rate(15)  # 4hz

n_drones = 5
running_size = 5

#-------------------

x_start = (1.5, 0, 0, 0, 0)
y_start = (0, 0.5, 0, 0, 0)
z_start = (0, 0, 0, -6, -8)

#----------------------

altitude = 1.3        #meters
robot_position = (0, 0, 0.72)
robot_radius = 0.7    #meters

#-----------------------------
# target_point = [(), ()]

class SendPositionNode(object):
    def __init__(self):
        self.msg = Path()
        self.pub = rospy.Publisher('drone1_rviz', Path, queue_size=10)

        self.gesture_number_right = 0
        self.gesture_position_right = np.zeros(3)
        self.gesture_number_left = 0
        self.gesture_position_left = np.zeros(3)

        self.pos_r = np.zeros(3)
        self.pos_l = np.zeros(3)

        self.gesture_number = 0
        self.gesture_position = np.zeros(3)

        self.n_drones = n_drones
        self.name_drones = []
        self.current_coordinates = []
        self.new_coordinates = []
        self.end_coordinates = []
        self.data1 = PoseStamped()
        self.data2 = PoseStamped()
        self.data3 = PoseStamped()
        self.data4 = PoseStamped()
        self.data5 = PoseStamped()
        self.catch_position_pub = PoseStamped()

        self.position_pattern = []
        self.count = 0

        self.intersection = ''

        self.previous_gesture_right = 99
        self.previous_gesture_left = 99

        self.status = 'BEGIN'
        self.add_hand = False

        self.swarm_config_r = 0
        self.swarm_config_l = 0

        self.hand_r_param_l = 0.5
        self.hand_r_param_a = 0

        self.hand_l_param_l = 0.5
        self.hand_l_param_a = 0

        self.delta_a = 0
        self.figure = 1

        time.sleep(1)
        self.publish_add_hund()

        self.collected_gesture = deque([0])
        for i in range(running_size - 1):
            self.collected_gesture.append(i)

        self.read_traj = []
        self.read_trak_all = []
        self.last_coordinates = []

        self.catch_position = []


    def plot_traj(self, true_traj):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        x_t = []
        y_t = []
        for i in range(len(true_traj)):
            x_t.append(true_traj[i][0])
            y_t.append(true_traj[i][1])

        ax.scatter(robot_position[0], robot_position[1], robot_position[2])
        ax.scatter(self.catch_position[0], self.catch_position[1], altitude)

        ax.plot3D(x_t, y_t, altitude, 'gray')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 3)
        plt.show()

    def init_position(self):
        for i in range(self.n_drones):
            self.current_coordinates.append([x_start[i], y_start[i], z_start[i]])
        for i in range(self.n_drones):
            self.end_coordinates.append([x_start[i], y_start[i], z_start[i]])
        print('init_position:', self.current_coordinates)
        print('end_position:', self.end_coordinates)

    def publish_add_hund(self):
        pub_add_hand.publish(str(self.add_hand))

    def callback_hand_right(self, msg):
        self.gesture_number_right = int(msg.pose.orientation.w)
        self.gesture_position_right = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.gesture_number = int(msg.pose.orientation.w)
        self.gesture_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def callback_hand_left(self, msg):
        self.gesture_number_left = int(msg.pose.orientation.w)
        self.gesture_position_left = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def get_end_poses(self):
        if self.status == 'BEGIN':
            self.end_coordinates = []
            for i in range(self.n_drones):
                self.end_coordinates.append([x_start[i], y_start[i], z_start[i]])

        if self.status == 'TAKE OFF':
            self.end_coordinates = []
            for i in range(self.n_drones):
                self.end_coordinates.append([x_start[i], y_start[i], z_start[i] + 1])
            if self.current_coordinates[0][2] == self.end_coordinates[0][2]:
                self.status = 'MODE: 1'

        if self.status == 'LAND':
            print('LAND')
            self.status = 'BEGIN'


            # self.end_coordinates = self.read_traj[len(self.read_traj) - 1]
            # print(self.end_coordinates)
            # for i in range(1):
            #     self.end_coordinates[i][2] = 0
            #     print(self.end_coordinates)
            # if self.current_coordinates[0][2] == self.end_coordinates[0][2]:
            #     self.status = 'BEGIN'

        if self.status == 'IN FLIGHT. MAIN':
            # print('we are here')
            self.end_coordinates = []
            for i in range(self.n_drones):
                self.end_coordinates.append([x_start[i], y_start[i], z_start[i] + 1])

        if self.status == 'TRAJECTORY':
            start = time.time()
            k = 0
            self.read_trak_all = []
            while k != len(self.read_traj):
                # -convert coordinates-
                frame_size = (640, 480, 50)
                shift = (100, 0.5, 0)
                room_size = (8, 3, 2.5)  # x, y, z

                room_shift = 1
                # transform px to coordinates VICON system (meters)
                # def transform(pos, frame_size, shift, room_size):
                #     new_coordinates = [(pos[0] / frame_size[0] - shift[1]) * room_size[1] + room_shift,
                #                        ((frame_size[1] - pos[1]) / frame_size[1] + shift[2]) * room_size[2]]
                #     return new_coordinates

                def transform(pos, frame_size, shift, room_size):
                    new_coordinates = [((pos[0] / 160) - 1.3) * 1.5, pos[1]/213 - 1.5]
                    return new_coordinates

                self.read_traj_meter = transform(self.read_traj[k], frame_size, shift, room_size)

                # self.prepare_data_out_and_publish(msg=' ')
                self.data1.pose.position.x = self.read_traj_meter[0]
                self.data1.pose.position.y = self.read_traj_meter[1]
                self.data1.pose.position.z = altitude

                # print(self.data1)

                if time.time() - start > 0.25:
                    start = time.time()
                    k += 1
                    pub1.publish(self.data1)
                    self.read_trak_all.append(self.read_traj_meter)

                    if k == len(self.read_traj) - 1:
                        self.status = 'IN FLIGHT. MAIN'
                        print('enddddd')
                        self.status = 'LAND'
                        print(self.read_trak_all)

                        self.plot_traj(self.read_trak_all)
                        #
                        # self.end_coordinates = []
                        # for i in range(self.n_drones):
                        #     self.end_coordinates.append([x_start[i], y_start[i], z_start[i] + 1])
                        # break

        # if self.status == 'MODE: 1':
        #
        #     if self.gesture_number_right == 5 or self.gesture_number_right == 4:
        #         self.last_coordinates = []
        #
        #         l = (self.hand_r_param_l * 0.5) * 0.5
        #         a = (self.delta_a - 90) / 2
        #         pi = math.pi
        #
        #         if self.figure == 1:
        #             x = (-l * math.cos(pi / 4 + a * pi / 180), 0,
        #                  l * math.cos(pi / 4 + a * pi / 180), 0, 0)
        #             y = (l * math.sin(pi / 4 + a * pi / 180), 0,
        #                  -l * math.sin(pi / 4 + a * pi / 180), 0, 0)
        #             z = (0, 0, 0, -2, -4)
        #
        #         if self.figure == 2:
        #             l -= 0.1
        #             x = (l * math.cos(7*pi / 6 - a * pi / 180), l * math.cos(11*pi / 6 - a * pi / 180),
        #                  l * math.cos(pi / 2 - a * pi / 180), 0, 0)
        #             y = (l * math.sin(7*pi / 6 - a * pi / 180), l * math.sin(11*pi / 6 - a * pi / 180),
        #                  l * math.sin(pi / 2 - a * pi / 180), 0, 0)
        #             z = (0, 0, 0, -6, -8)
        #
        #         # x = (-l * math.cos(pi / 4 + a * pi / 180), l * math.cos(pi / 4 - a * pi / 180),
        #         #      l * math.cos(pi / 4 + a * pi / 180), -l * math.cos(pi / 4 - a * pi / 180), 0)
        #         # y = (l * math.sin(pi / 4 + a * pi / 180), l * math.sin(pi / 4 - a * pi / 180),
        #         #      -l * math.sin(pi / 4 + a * pi / 180), -l * math.sin(pi / 4 - a * pi / 180), 0)
        #         # z = (0, 0, 0, 0, 0)
        #
        #         # x = (-l, l, l, -l, 0)
        #         # y = (l, l, -l, -l, 0)
        #         # z = (0, 0, 0, 0, 0)
        #
        #     self.end_coordinates = []
        #     for i in range(self.n_drones):
        #         self.end_coordinates.append(
        #             [self.pos_r[0] + x[i], self.pos_r[1] + y[i], self.pos_r[2] + z[i]])
        #         self.last_coordinates.append(
        #             [self.pos_r[0] + x[i], self.pos_r[1] + y[i], self.pos_r[2] + z[i]])
        #     # print('end_coordinates', self.end_coordinates)

    def move_to_end(self, msg):
        # ---function to move drones from CURRENT to END positions---

        # ---Coefitiont of potential fields---
        # save distance between drones
        r_save = 0.35
        # finish radius of scan
        r_scan = 0.1
        # length of total_vector
        dt = 0.02
        coef_obsticl = 30
        # ---End coefitiones---

        # ---function to set end/next positons for swarm---
        self.get_end_poses()

        # ----FUNCTIONS----
        def make_vectors_by_points(points_start, points_end):
            k = len(points_start)
            vector = np.zeros(shape=(k, 3))
            norm = np.zeros(shape=(k, 3))
            dist = points_start - points_end
            for i in range(k):
                norm[i] = math.sqrt(dist[i, 0] ** 2 + dist[i, 1] ** 2 + dist[i, 2] ** 2)
                if norm[i].all() != 0:
                    vector[i] = dist[i] / norm[i]
                else:
                    vector[i] = 0
            return vector

        def make_vector_two_points(point1, point2):
            dist = point1 - point2
            norm = (dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) ** (0.5)
            vector = np.zeros(3)
            if norm.all() > 0.0001:
                vector = dist / norm
            return vector

        def calculate_collision_coef(r_save, point1, point2):
            dist = point1 - point2
            dist = (dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) ** (0.5)
            if dist >= r_save:
                koef = 0
            else:
                koef = (r_save - dist) ** 2
            return (koef)

        def get_distance_between_points(point1, point2):
            k = len(point1)
            dist = np.zeros(k)
            dp = point1 - point2
            for i in range(k):
                dist[i] = math.sqrt(dp[i, 0] ** 2 + dp[i, 1] ** 2 + dp[i, 2] ** 2)
            return dist

        # ----END FUNCTIONS----
        currents = np.array(self.current_coordinates)
        ends = np.array(self.end_coordinates)

        # print('current', self.current_coordinates)
        # print('end', self.end_coordinates)

        # --start moove--
        vector_target = make_vectors_by_points(currents, ends)
        vector_total = vector_target

        # print('vector_target', vector_target)

        # Calculating obsticalte vectors
        coefficient_obstical = np.zeros(self.n_drones - 1)
        vector_obstical = np.zeros(shape=((self.n_drones - 1), 3))

        for i in range(self.n_drones):
            for j in range(self.n_drones):
                if i != j:
                    coefficient_obstical[j - 1] = calculate_collision_coef(r_save, currents[i], currents[j])
                    vector_obstical[j - 1] = make_vector_two_points(currents[i], currents[j]) * coefficient_obstical[
                        j - 1]
                    vector_total[j] += vector_obstical[j - 1] * coef_obsticl

        current_point = currents - dt * vector_total
        check_distance = get_distance_between_points(current_point, ends)
        # print(check_distance)

        for i in range(self.n_drones):
            if check_distance[i] <= r_scan:
                self.current_coordinates[i] = ends[i]
            else:
                self.current_coordinates[i] = current_point[i]

        # print(self.current_coordinates - ends)
        # self.current_coordinates = current_point

    def subscriber_gesture(self):
        rospy.Subscriber('/hand_position_right', PoseStamped, self.callback_hand_right)
        rospy.Subscriber('/hand_position_left', PoseStamped, self.callback_hand_left)
        rospy.Subscriber('/hands_boarder', String, self.callback_hands_boarder)
        rospy.Subscriber('/letter_drone', String, self.resent)
        rospy.Subscriber('/hands_parameters', String, self.get_hand_parameters)
        rospy.Subscriber('/send_traj', String, self.start_flight)

    def start_flight(self, msg):
        # self.read_traj_mouse_csv_true()
        # time.sleep(1)
        self.status = 'TRAJECTORY'
        print('TRAJECTORY. mode')


    def get_catch_position(self):
        x = []
        y = []
        for i in range(len(self.read_trak_all)):
            x.append(self.read_trak_all[i][0])
            y.append(self.read_trak_all[i][1])

        x = [abs(ele) for ele in x]

        min_value = min(x)
        min_index = x.index(min_value)
        # print('min')
        # print(min_value, min_index)

        dist = math.sqrt( (robot_position[0] - x[min_index]) ** 2 +
                          (robot_position[1] - y[min_index]) ** 2 +
                          (robot_position[2] - altitude) ** 2)

        self.catch_position = (x[min_index], y[min_index])
        print(dist)

        self.catch_position_pub.pose.position.x = self.catch_position[0]
        self.catch_position_pub.pose.position.y = -self.catch_position[1]
        self.catch_position_pub.pose.position.z = altitude

        if (dist < robot_radius):

            print("catch_position" + str(self.catch_position))
            pub_catch_pos.publish(self.catch_position_pub)
        else:
            print('cant catch')

    def resent(self, msg):
        pub_for_drawing.publish(msg.data)

    def callback_hands_boarder(self, msg):
        self.intersection = msg.data
        # print(self.intersection)

    def coordinates_processsing(self, msg):
        # --- function to transform screnn coordinates of the hands           ---
        # --- to coordinates for drones in 'mocap' room with size 'room_size' ---
        # |---------|
        # |          exit
        # |      y> |
        # |   xv    |
        # |       pc|
        # |---------|

        frame_size = (640, 480, 50)
        shift = (25, 0.5, 0)
        room_size = (3, 3, 2.5)  # x, y, z

        def transform(pos, frame_size, shift, room_size):
            new_coordinates = [(frame_size[2] - pos[2] - shift[0]) / frame_size[2] * room_size[0],
                               (pos[0] / frame_size[0] - shift[1]) * room_size[1],
                               ((frame_size[1] - pos[1]) / frame_size[1] + shift[2]) * room_size[2]]
            return new_coordinates

        # x(size of palm, deep), y(left-right), z(up-down)
        # frame size x:640, y:480
        # close [31.26143766018474, 365.0220321347224, 169.18150273900386]
        # center [20.701394586524664, 349.1738313342903, 188.26702847228626]
        # deep [10.391559272269477, 341.6072448182312, 230.87777361631385]

        self.pos_r = transform(self.gesture_position_right, frame_size, shift, room_size)
        self.pos_l = transform(self.gesture_position_left, frame_size, shift, room_size)

        # print(self.pos_r, self.pos_l)
        # self.plot_coordinates()

        # ---transform angle of the hand. Gesture 5---
        self.delta_a += int(self.hand_r_param_a / 20) * 1.5
        # print(self.delta_a)

    def get_hand_parameters(self, msg):
        data = msg.data
        hand_parameters = re.findall(r"[-+]?\d*\.\d+|\d+", data)

        # print(hand_parameters)
        if self.gesture_number_right == 5:
            self.hand_r_param_l = float(hand_parameters[2])
            self.hand_r_param_a = float(hand_parameters[3])
        if self.gesture_number_left == 5:
            self.hand_l_param_l = float(hand_parameters[0])
            self.hand_l_param_a = float(hand_parameters[1])
        # print(self.hand_r_param_l, self.hand_r_param_a)

        # print(int(self.hand_r_param_a/15), self.delta_a)
        # print(hand_parameters)

    def subscriber_clock(self):
        # ---subscriper for trajectory generation--
        rospy.Subscriber('/test_clock_py', String, self.coordinates_processsing)
        # rospy.Subscriber('/test_clock_py', String, self.gesture_system_control)  # swarm_hand control
        rospy.Subscriber('/test_clock_py', String, self.gesture_control_send_traj)  # drawing trajectory
        # rospy.Subscriber('/test_clock_py', String, self.gesture_system_control_land_take_off)
        rospy.Subscriber('/test_clock_py', String, self.prepare_data_out_and_publish)

        if self.end_coordinates != []:
            rospy.Subscriber('/test_clock_py', String, self.move_to_end)

    def read_traj_csv_true(self):
        print('read_traj')
        with open('/home/sk/PycharmProjects/hand_tracking-master-2/drone_trap/draw_drone.csv', newline='') as file:
            true_traj = []
            data = csv.reader(file)
            for row in data:
                row = np.array(row)
                for i in range(len(row)):
                    true_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", true_traj[i])
                    true_traj[i] = (float(data_float[1]), float(data_float[0]))
                self.read_traj = true_traj
            print(self.read_traj)

        def transform(pos):
            new_coordinates = [((pos[0] / 160) - 1.3) * 1.5, pos[1] / 213 - 1.5]
            return new_coordinates

        self.read_trak_all = []
        k = 0
        while k != len(self.read_traj):
            self.read_traj_meter = transform(self.read_traj[k])
            self.read_trak_all.append(self.read_traj_meter)
            k += 1

        self.get_catch_position()
        print(self.read_trak_all)


    def read_traj_mouse_csv_true(self):
        with open('/home/sk/PycharmProjects/hand_tracking-master-2/trajectories/mouse_drone.csv', newline='') as file:
            true_traj = []
            data = csv.reader(file)
            for row in data:
                row = np.array(row)
                for i in range(len(row)):
                    true_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", true_traj[i])
                    true_traj[i] = (float(data_float[0]), float(data_float[1]))
                self.read_traj = true_traj
            print(self.read_traj)

    def gesture_control_send_traj(self, msg):
        # ----moving_average, get smooth data about number of gesture----
        self.collected_gesture.rotate(1)
        self.collected_gesture[0] = self.gesture_number_right
        average = sum(self.collected_gesture) / running_size
        rounding = round(average)
        identical = 1
        for i in range(len(self.collected_gesture)):
            for j in range(len(self.collected_gesture)):
                if self.collected_gesture[i] == self.collected_gesture[j]:
                    identical = identical * 1
                else:
                    identical = identical * 0
        # print(self.collected_gesture, average, identical)

        current_gesture_right = self.gesture_number_right
        current_gesture_left = self.gesture_number_left

        if (current_gesture_right != self.previous_gesture_right) and identical:
            print(self.status)
            self.count += 1
            # print(self.collected_gesture, average, rounding, identical)
            print(self.count, 'previous:', self.previous_gesture_right, ', current:', current_gesture_right)

            if (self.status != 'BEGIN') and (self.previous_gesture_right == 5) and (current_gesture_right == 8):
                self.status = 'BEGIN'
                print('LAND ALL')
                pub_for_drawing.publish('land')
                time.sleep(2)

            if (self.status == 'BEGIN') and (self.previous_gesture_right == 1) and (current_gesture_right == 8):
                print('From begin to TAKE OFF')
                self.status = 'TAKE OFF'
                pub_for_drawing.publish('takeoff')
                time.sleep(2)

            # ---END FLIGHT---
            if (self.status == 'IN FLIGHT. MAIN') and (self.previous_gesture_right == 5) and (
                    current_gesture_right == 8):
                self.status = 'LAND'
                print('LAND')
                pub_for_drawing.publish('land')
                time.sleep(2)

            # ---READ AND SEND TRAJECTORIES---
            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (
                    current_gesture_right == 6):
                print('READ TRAJ')
                time.sleep(2)
                self.read_traj_csv_true()
                self.status = 'TRAJECTORY'

            self.previous_gesture_right = current_gesture_right
            self.previous_gesture_left = current_gesture_left

    def gesture_system_control(self, msg):
        # ----moving_average, get smooth data about number of gesture----
        self.collected_gesture.rotate(1)
        self.collected_gesture[0] = self.gesture_number_right
        average = sum(self.collected_gesture) / running_size
        rounding = round(average)
        identical = 1
        for i in range(len(self.collected_gesture)):
            for j in range(len(self.collected_gesture)):
                if self.collected_gesture[i] == self.collected_gesture[j]:
                    identical = identical * 1
                else:
                    identical = identical * 0
        # print(self.collected_gesture, average, identical)

        current_gesture_right = self.gesture_number_right
        current_gesture_left = self.gesture_number_left

        if (current_gesture_right != self.previous_gesture_right) and identical:
            self.count += 1
            # print(self.collected_gesture, average, rounding, identical)
            print(self.count, 'previous:', self.previous_gesture_right, ', current:', current_gesture_right)

            if (self.status != 'BEGIN') and (self.previous_gesture_right == 5) and (current_gesture_right == 8):
                self.status = 'BEGIN'
                print('LAND ALL')
                pub_for_drawing.publish('land')
                time.sleep(2)

            if (self.status == 'BEGIN') and (self.previous_gesture_right == 1) and (current_gesture_right == 8):
                print('From begin to TAKE OFF')
                # self.status = 'MODE: 1'
                self.status = 'IN FLIGHT. MAIN'
                # self.status = 'LAND'
                pub_for_drawing.publish('takeoff')
                time.sleep(2)

            # ---END FLIGHT---
            if (self.status == 'IN FLIGHT. MAIN') and (self.previous_gesture_right == 5) and (
                    current_gesture_right == 8):
                self.status = 'LAND'
                print('LAND')
                pub_for_drawing.publish('land')
                time.sleep(2)

            # ---CHOOSE MODE 1. CHANGE FORMS OF SWARM---
            if (self.status == 'IN FLIGHT. MAIN') and (self.previous_gesture_right == 5) and (
                    current_gesture_right == 1):
                print('MODE 1: CHANGE FORMS OF SWARM')
                self.status = 'MODE: 1'

            # # ---CHOOSE MODE 2. TWO HAND CONTROL---
            # if (self.status == 'IN FLIGHT. MAIN') and (self.previous_gesture_right == 5) and (
            #         current_gesture_right == 2):
            #     print('MODE 2: SPLIT/MERGE SWAMRS')
            #     self.status = 'TWO HAND CONTROL, ONE SWARM'
            #
            # # ---MODE 2. SPLIT, MAKE TWO SWARMS---
            # if (self.status == 'TWO HAND CONTROL, ONE SWARM') and (self.intersection == 'Intersect') and (
            #         self.previous_gesture_right == 5) and (
            #         (current_gesture_right == 1) or (current_gesture_right == 2) or (current_gesture_right == 3)):
            #     self.status = 'TWO HAND CONTROL, TWO SWARM'
            #     print(
            #         'swarm splited,' + str(current_gesture_left) + ' drones, ' + str(current_gesture_right) + ' drones')
            #
            # # ---MODE 2. MERGE, RETURN TO ONE SWARM---
            # if (self.status == 'TWO HAND CONTROL, TWO SWARM') and (self.intersection == 'Intersect') and (
            #         self.previous_gesture_right == 5) and (current_gesture_right == 6) and (current_gesture_left == 6):
            #     self.status = 'TWO HAND CONTROL, ONE SWARM'
            #     print('swarm merged')
            #
            # # ---CHOOSE MODE 2. RETURN TO THE MAIN MENU---
            # if (self.status == 'TWO HAND CONTROL, ONE SWARM') and (self.previous_gesture_right == 5) and (
            #         current_gesture_right == 6):
            #     print('RETURN TO THE MAIN MENU')
            #     self.status = 'IN FLIGHT. MAIN'

            # ---MODE 1---
            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 1):
                print('fig ONE')
                self.figure = 1
            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 2):
                print('fig TWO')
                self.figure = 2
            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 3):
                print('fig THREE')
                self.figure = 3
            # if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 4):
            #     print('fig FORE')
            #     self.figure = 4
            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 6):
                print('IN FLIGHT. MAIN')
                self.status = 'IN FLIGHT. MAIN'

            self.previous_gesture_right = current_gesture_right
            self.previous_gesture_left = current_gesture_left

    def gesture_system_control_land_take_off(self, msg):
        # ----moving_average, get smooth data about number of gesture----
        self.collected_gesture.rotate(1)
        self.collected_gesture[0] = self.gesture_number_right
        average = sum(self.collected_gesture) / running_size
        rounding = round(average)
        identical = 1
        for i in range(len(self.collected_gesture)):
            for j in range(len(self.collected_gesture)):
                if self.collected_gesture[i] == self.collected_gesture[j]:
                    identical = identical * 1
                else:
                    identical = identical * 0
        # print(self.collected_gesture, average, identical)

        current_gesture_right = self.gesture_number_right
        current_gesture_left = self.gesture_number_left

        if (current_gesture_right != self.previous_gesture_right) and identical:
            self.count += 1
            # print(self.collected_gesture, average, rounding, identical)
            print(self.count, 'previous:', self.previous_gesture_right, ', current:', current_gesture_right)

            if (self.status == 'BEGIN') and (self.previous_gesture_right == 2) and (current_gesture_right == 8):
                self.status = 'TAKE OFF'
                # print(self.status)
                pub_for_drawing.publish('takeoff')
                time.sleep(2)

                self.status = 'MODE: 1'

            if (self.status == 'MODE: 1') and (self.previous_gesture_right == 5) and (current_gesture_right == 8):
                self.status = 'BEGIN'
                print("LAND ALL")
                pub_for_drawing.publish('land')
                time.sleep(2)

            self.previous_gesture_right = current_gesture_right
            self.previous_gesture_left = current_gesture_left

    def prepare_data_out_and_publish(self, msg):
        # drone1
        self.data1.pose.position.x = self.current_coordinates[0][0]
        self.data1.pose.position.y = self.current_coordinates[0][1]
        self.data1.pose.position.z = self.current_coordinates[0][2]
        # drone2
        self.data2.pose.position.x = self.current_coordinates[1][0]
        self.data2.pose.position.y = self.current_coordinates[1][1]
        self.data2.pose.position.z = self.current_coordinates[1][2]
        # drone3
        self.data3.pose.position.x = self.current_coordinates[2][0]
        self.data3.pose.position.y = self.current_coordinates[2][1]
        self.data3.pose.position.z = self.current_coordinates[2][2]
        # drone4
        self.data4.pose.position.x = self.current_coordinates[3][0]
        self.data4.pose.position.y = self.current_coordinates[3][1]
        self.data4.pose.position.z = self.current_coordinates[3][2]
        # drone5
        self.data5.pose.position.x = self.current_coordinates[4][0]
        self.data5.pose.position.y = self.current_coordinates[4][1]
        self.data5.pose.position.z = self.current_coordinates[4][2]
        # publish positions
        self.publish_positions()

    def publish_positions(self):
        pub1.publish(self.data1)
        pub2.publish(self.data2)
        pub3.publish(self.data3)
        pub4.publish(self.data4)
        pub5.publish(self.data5)
        self.publish((self.current_coordinates[0][0],
                      self.current_coordinates[0][1],
                      self.current_coordinates[0][2]), 0)
        rate.sleep()

    def msg_def_path(self, point, yaw):
        worldFrame = rospy.get_param("~worldFrame", "/world")
        self.msg.header.stamp = rospy.Time.now()
        self.msg.header.frame_id = worldFrame
        pose = PoseStamped()
        pose.pose.position.x = point[0]
        pose.pose.position.y = point[1]
        pose.pose.position.z = point[2]
        self.msg.poses.append(pose)
        return self.msg

    def publish(self, data, yaw=0):
        msg = self.msg_def_path(data, yaw)
        self.pub.publish(msg)
        #rospy.Rate(100).sleep()
        return


if __name__ == '__main__':
    try:
        node = SendPositionNode()
        node.init_position()
        while True:
            node.subscriber_gesture()
            node.subscriber_clock()
            rospy.spin()


    except rospy.ROSInterruptException:
        pass
