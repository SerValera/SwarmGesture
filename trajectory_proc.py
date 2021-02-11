#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

import csv
import re
import matplotlib
import math

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- choose trajectory ---
number_traj = 1


class TrajProcessing(object):
    def __init__(self):
        self.true_traj = []
        self.draw_traj = []
        self.number_traj = number_traj
        self.n_draw = 0
        self.intrapolated_traj = []

        self.choose_traj = 2

    def read_traj_csv_true(self):
        with open('trajectories/true_traj_' + str(self.number_traj) + '.csv', newline='') as file:
            true_traj = []
            data = csv.reader(file)
            for row in data:
                row = np.array(row)
                for i in range(len(row)):
                    true_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", true_traj[i])
                    true_traj[i] = (float(data_float[0]), float(data_float[1]))
                self.true_traj = true_traj

    def read_traj_csv_drawing_all(self):
        with open('trajectories/draw_traj_' + str(self.number_traj) + '.csv', newline='') as file:
            data = csv.reader(file)
            self.n_draw = 0
            for row in data:
                self.n_draw += 1
                # print(self.n_draw)
                row = np.array(row)
                draw_traj = []
                for i in range(len(row)):
                    draw_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", draw_traj[i])
                    draw_traj[i] = (float(data_float[0]), float(data_float[1]))
                self.draw_traj.append(draw_traj)
            # for i in range(len(self.draw_traj)):
            # print(self.draw_traj[i])

    def plot_traj(self):

        def plot_t(traj):
            x = []
            y = []
            for i in range(len(traj)):
                x.append(self.draw_traj[k][i][0])
                y.append(self.draw_traj[k][i][1])
            plt.plot(x, y, label='Drawing traj' + str(k))
            plt.plot(x, y, 'bo', markersize=1)

        plt.figure()
        plt.title('Trajectories')
        plt.xlim([0, 640])
        plt.ylim([0, 480])

        x_t = []
        y_t = []
        for i in range(len(self.true_traj)):
            x_t.append(self.true_traj[i][0])
            y_t.append(self.true_traj[i][1])
        plt.plot(x_t, y_t, label='True trajectory')
        plt.plot(x_t, y_t, 'bo', markersize=2)

        x = []
        y = []
        if self.n_draw == 1:
            for i in range(len(self.draw_traj)):
                x.append(self.draw_traj[i][0])
                y.append(self.draw_traj[i][1])
                print(x, y)
            plt.plot(x, y, label='Drawing traj 1')

        # plot all
        # if self.n_draw > 1:
        #     for k in range(self.n_draw):
        #         x = []
        #         y = []
        #         for i in range(len(self.draw_traj[k])):
        #             x.append(self.draw_traj[k][i][0])
        #             y.append(self.draw_traj[k][i][1])
        #         plt.plot(x, y, label='Drawing traj' + str(k))

        # plot one drawn traj

        k = self.choose_traj
        x = []
        y = []
        for i in range(len(self.draw_traj[k])):
            x.append(self.draw_traj[k][i][0])
            y.append(self.draw_traj[k][i][1])
        plt.plot(x, y, label='Drawing traj' + str(k))
        plt.plot(x, y, 'ro', markersize=3)

        # x = []
        # y = []
        # for i in range(len(self.intrapolated_traj)):
        #     x.append(self.intrapolated_traj[i][0])
        #     y.append(self.intrapolated_traj[i][1])
        # plt.plot(x, y, label='Intrapol traj')
        # plt.plot(x, y, 'go', markersize=5)

        plt.legend()
        plt.show()


    def intrapolate(self):
        def interpcurve(N, pX, pY):
            # equally spaced in arclength
            N = np.transpose(np.linspace(0, 1, N))

            # how many points will be uniformly interpolated?
            nt = N.size

            # number of points on the curve
            n = pX.size
            pxy = np.array((pX, pY)).T
            p1 = pxy[0, :]
            pend = pxy[-1, :]
            last_segment = np.linalg.norm(np.subtract(p1, pend))
            epsilon = 10 * np.finfo(float).eps

            # IF the two end points are not close enough lets close the curve
            if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
                pxy = np.vstack((pxy, p1))
                nt = nt + 1
            else:
                print('Contour already closed')

            pt = np.zeros((nt, 2))

            # Compute the chordal arclength of each segment.
            chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
            # Normalize the arclengths to a unit total
            chordlen = chordlen / np.sum(chordlen)
            # cumulative arclength
            cumarc = np.append(0, np.cumsum(chordlen))

            tbins = np.digitize(N, cumarc)  # bin index in which each N is in

            # catch any problems at the ends
            tbins[np.where(tbins <= 0 | (N <= 0))] = 1
            tbins[np.where(tbins >= n | (N >= 1))] = n - 1

            s = np.divide((N - cumarc[tbins]), chordlen[tbins - 1])
            pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)
            return pt

        x, y = [], []
        for i in range(len(self.draw_traj[self.choose_traj])):
            x.append(self.draw_traj[self.choose_traj][i][0])
            y.append(self.draw_traj[self.choose_traj][i][1])

        x = np.array(x)
        y = np.array(y)

        print(x)
        print(y)

        self.intrapolated_traj = interpcurve(50, x, y)




if __name__ == '__main__':
    try:
        node = TrajProcessing()
        node.read_traj_csv_true()
        node.read_traj_csv_drawing_all()
        #node.intrapolate()
        node.plot_traj()

        while True:
            print('1')
            # rospy.spin()

    except rospy.ROSInterruptException:
        pass
