#!/usr/bin/env python
import rospy
import numpy as np
import csv
import re
import matplotlib
import math

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- choose participance and number of trajectory ---
land_position = 0
n_particapant = 6

color = ('red', 'green', 'blue', 'gray')
color_point = ('r^', 'g^', 'b^')
size_last_drones = 12
plot_target = ('black', '*', 15)
plot_center = ('gray', '^', 15)
plot_c_target = ('black', 'o', 8)
# -----------------------------------------------------

x_target = ([0, 0, 0], [0, 0, -0.433])
y_target = ([-0.4, 0, 0.4], [-0.25, 0.25, 0])
c_target = []

for i in range(len(x_target)):
    c_target.append([(y_target[i][0] + y_target[i][1] + y_target[i][2]) / 3,
                     (x_target[i][0] + x_target[i][1] + x_target[i][2]) / 3])


class TrajProcessing(object):
    def __init__(self):
        self.traj_0_drones = []
        self.traj_1_drones = []
        self.traj_center = []

        self.x_c_0 = []
        self.y_c_0 = []

        self.x_c_1 = []
        self.y_c_1 = []

        self.center_error = []

    def plot_traj(self, n_land):
        plt.figure()
        plt.title('Trajectories')
        plt.xlabel("X, meters")
        plt.ylabel("Y, meters")
        plt.xlim([-1.5, 1.5])
        plt.xlim([-0.75, 0.75])

        # plot target points. black color
        for i in range(3):
            plt.plot(y_target[n_land][i], x_target[n_land][i], color=plot_target[0], marker=plot_target[1],
                     markersize=plot_target[2], label='Target point')

        # plot center of mass target point
        plt.plot(c_target[n_land][0], c_target[n_land][1], color=plot_c_target[0], marker=plot_c_target[1],
                 markersize=plot_c_target[2], label='Center')

        if n_land == 0:
            for i in range(3):
                plt.plot(self.traj_0_drones[i][:, 1], self.traj_0_drones[i][:, 0], color=color[i], linewidth=1.0,
                         linestyle='--', label='Drone path' + str(i + 1))

                n_last = len(self.traj_0_drones[i]) - 1
                plt.plot(self.traj_0_drones[i][n_last, 1], self.traj_0_drones[i][n_last, 0], color_point[i],
                         markersize=size_last_drones, label='Drone land point')

            plt.plot(self.x_c_0, self.y_c_0, color=plot_center[0], linewidth=1.0, linestyle='--',
                     label='Center of swarm')
            plt.plot(self.x_c_0[len(self.x_c_0) - 1], self.y_c_0[len(self.y_c_0) - 1], color=plot_center[0],
                     marker=plot_center[1], markersize=plot_center[2], label='center of the swarm')

        if n_land == 1:
            for i in range(3):
                plt.plot(self.traj_1_drones[i][:, 1], self.traj_1_drones[i][:, 0], color=color[i], linewidth=1.0,
                         linestyle='--', label='Drone path' + str(i + 1))

                n_last = len(self.traj_1_drones[i]) - 1
                plt.plot(self.traj_1_drones[i][n_last, 1], self.traj_1_drones[i][n_last, 0], color_point[i],
                         markersize=size_last_drones)

            plt.plot(self.x_c_1, self.y_c_1, color=color[3], linewidth=1.0, linestyle='--', label='Center of swarm')
            plt.plot(self.x_c_1[len(self.x_c_1) - 1], self.y_c_1[len(self.y_c_1) - 1], color=plot_center[0],
                     marker=plot_center[1], markersize=plot_center[2])

        plt.gca().invert_yaxis()
        #plt.grid()
        plt.legend()
        plt.show()

    def read_drones_traj(self):
        def read_from_csv(path):
            with open(path, 'r') as f:
                data = list(csv.reader(f))[0]
            for i in range(len(data)):
                data[i] = data[i][1:-1].split()
                for j in range(len(data[i])):
                    if data[i][j][-1] == ',':
                        data[i][j] = float(data[i][j][:-1])
                    else:
                        data[i][j] = float(data[i][j])
            return np.array(data)

        for j in range(3):
            filename = ('traj_unity/p_' + str(n_particapant) + '_land_n_' + str(0) + '_traj_unity_' + str(
                j + 1) + '.csv')
            self.traj_0_drones.append(read_from_csv(filename))

        for j in range(3):
            filename = ('traj_unity/p_' + str(n_particapant) + '_land_n_' + str(1) + '_traj_unity_' + str(
                j + 1) + '.csv')
            self.traj_1_drones.append(read_from_csv(filename))

    def make_center_mass_traj(self, n_land):
        points_traj = []

        if n_land == 0:
            for i in range(3):
                points_traj.append(len(self.traj_0_drones[i]))

        if n_land == 1:
            for i in range(3):
                points_traj.append(len(self.traj_1_drones[i]))

        n_points = min(points_traj)

        x_c = []
        y_c = []

        if n_land == 0:
            for j in range(n_points):
                y_c.append(
                    (self.traj_0_drones[0][j, 0] + self.traj_0_drones[1][j, 0] + self.traj_0_drones[2][j, 0]) / 3)
                x_c.append(
                    (self.traj_0_drones[0][j, 1] + self.traj_0_drones[1][j, 1] + self.traj_0_drones[2][j, 1]) / 3)

            self.x_c_0 = np.array(x_c)
            self.y_c_0 = np.array(y_c)

        if n_land == 1:
            for j in range(n_points):
                y_c.append(
                    (self.traj_1_drones[0][j, 0] + self.traj_1_drones[1][j, 0] + self.traj_1_drones[2][j, 0]) / 3)
                x_c.append(
                    (self.traj_1_drones[0][j, 1] + self.traj_1_drones[1][j, 1] + self.traj_1_drones[2][j, 1]) / 3)
            self.x_c_1 = np.array(x_c)
            self.y_c_1 = np.array(y_c)

    def get_center_error(self):
        error = math.sqrt((self.x_c_0[len(self.x_c_0) - 1] - c_target[0][0]) ** 2 + (self.y_c_0[len(self.y_c_0) - 1] - c_target[0][1]) ** 2)
        self.center_error.append(error)

        error = math.sqrt((self.x_c_1[len(self.x_c_1) - 1] - c_target[1][0]) ** 2 + (self.y_c_1[len(self.y_c_1) - 1] - c_target[1][1]) ** 2)
        self.center_error.append(error)

        print(self.center_error)


if __name__ == '__main__':
    try:
        node = TrajProcessing()
        node.read_drones_traj()

        node.make_center_mass_traj(0)
        node.make_center_mass_traj(1)

        node.plot_traj(0)
        node.plot_traj(1)

        node.get_center_error()


    except rospy.ROSInterruptException:
        pass
