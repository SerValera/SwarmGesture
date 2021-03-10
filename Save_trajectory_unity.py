#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import csv

import math

import re
import tkinter
import matplotlib
import time

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

rospy.init_node('unity_trajectories', anonymous=True)
rate = rospy.Rate(20)  # 10hz

participant = 99
land_position = 1

x_target = ([0, 0, 0], [0, 0, -0.433])
y_target = ([-0.4, 0, 0.4], [-0.25, 0.25, 0])

# z_start = (0, 0, 0)
class RecordTraj(object):
    def __init__(self):
        self.traj_dron1 = []
        self.traj_dron2 = []
        self.traj_dron3 = []
        self.status = 0
        self.last_coord = np.zeros(shape=(3, 2))
        self.delta_time = 0
        self.erros_drones = []

    def callback_1(self, msg):
        if self.status == 1:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            self.traj_dron1.append([x, y, z])

    def callback_2(self, msg):
        if self.status == 1:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            self.traj_dron2.append([x, y, z])

    def callback_3(self, msg):
        if self.status == 1:
            x = msg.pose.position.x
            y = msg.pose.position.y
            z = msg.pose.position.z
            self.traj_dron3.append([x, y, z])

    def save_traj_csv(self, traj, number):
        with open('traj_unity/p_' + str(participant) + '_land_n_' + str(land_position) + '_traj_unity_' + str(number) + '.csv', 'a') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(traj)

    def plot_trajectory(self):

        plt.figure()
        plt.title('Trajectories')
        x_1 = []
        y_1 = []
        x_2 = []
        y_2 = []
        x_3 = []
        y_3 = []

        for i in range(len(self.traj_dron1)):
            x_1.append(self.traj_dron1[i][0])
            y_1.append(self.traj_dron1[i][1])

        for i in range(len(self.traj_dron2)):
            x_2.append(self.traj_dron2[i][0])
            y_2.append(self.traj_dron2[i][1])

        for i in range(len(self.traj_dron3)):
            x_3.append(self.traj_dron3[i][0])
            y_3.append(self.traj_dron3[i][1])

        color = ('red', 'green', 'blue')
        color_point = ('ro', 'go', 'bo')

        for i in range(3):
            plt.plot(y_target[land_position][i], x_target[land_position][i], color_point[i], markersize=5)

        self.last_coord[0, 0] = self.traj_dron1[len(self.traj_dron1) - 1][1]
        self.last_coord[0, 1] = self.traj_dron1[len(self.traj_dron1) - 1][0]
        self.last_coord[1, 0] = self.traj_dron2[len(self.traj_dron2) - 1][1]
        self.last_coord[1, 1] = self.traj_dron2[len(self.traj_dron2) - 1][0]
        self.last_coord[2, 0] = self.traj_dron3[len(self.traj_dron3) - 1][1]
        self.last_coord[2, 1] = self.traj_dron3[len(self.traj_dron3) - 1][0]

        self.get_erros()
        self.save_time_csv()

        for i in range(3):
            plt.plot(self.last_coord[i, 0], self.last_coord[i, 1], color_point[i], markersize=4)

        plt.plot(y_1, x_1, color=color[0], linewidth=1.0, linestyle='--', label='Drone path 1')
        plt.plot(y_2, x_2, color=color[1], linewidth=1.0, linestyle='--', label='Drone path 2')
        plt.plot(y_3, x_3, color=color[2], linewidth=1.0, linestyle='--', label='Drone path 3')
        plt.xlabel("X, meters")
        plt.ylabel("Y, meters")
        plt.xlim([-1.5, 1.5])
        plt.xlim([-1, 1])

        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()
        plt.show()

    def get_erros(self):
        dist = []
        for i in range(3):
            drone_land = (self.last_coord[i, 0], self.last_coord[i, 1])
            target = (y_target[land_position][i], x_target[land_position][i])
            dist.append(math.sqrt((drone_land[0] - target[0]) ** 2 + (drone_land[1] - target[1]) ** 2))
            print('error drone #' + str(i), dist[i])

        self.erros_drones = dist


    def swarm_control(self, msg):
        global start
        data = msg.data
        print(data)
        if data == 'takeoff':
            self.status = 1
            self.traj_dron1 = []
            self.traj_dron2 = []
            self.traj_dron3 = []
            d_time = 0
            start = time.time()
            time.perf_counter()

        if data == 'land':
            self.erros_drones = []
            self.traj_dron1_ar = np.array(self.traj_dron1)
            self.traj_dron2_ar = np.array(self.traj_dron2)
            self.traj_dron3_ar = np.array(self.traj_dron3)

            self.save_traj_csv(self.traj_dron1_ar, 1)
            self.save_traj_csv(self.traj_dron2_ar, 2)
            self.save_traj_csv(self.traj_dron3_ar, 3)

            end_time = time.time()
            self.delta_time = end_time - start

            print('Your time:', self.delta_time)

            self.plot_trajectory()

            self.status = 0

    def save_time_csv(self):
        with open('traj_unity/p_' + str(participant) + '_land_n_' + str(land_position) + '_traj_unity_dtime.csv', 'a') as f:
            thewriter = csv.writer(f)
            data = []
            data.append(self.delta_time)

            for i in range(3):
                data.append(self.erros_drones[i])

            thewriter.writerow(data)

    def subscriber_drone_unity(self):
        rospy.Subscriber('/drone1u', PoseStamped, self.callback_1)
        rospy.Subscriber('/drone2u', PoseStamped, self.callback_2)
        rospy.Subscriber('/drone3u', PoseStamped, self.callback_3)
        rospy.Subscriber('/drone_drawing', String, self.swarm_control)


if __name__ == '__main__':
    try:
        node = RecordTraj()
        while True:
            node.subscriber_drone_unity()
            rospy.spin()


    except rospy.ROSInterruptException:
        pass
