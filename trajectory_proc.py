#!/usr/bin/env python
import rospy
import numpy as np
import csv
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# --- choose participance and number of trajectory ---
number_traj = 1
n_particapant = 3

names_traj = ('square', 'circle', 'triangle')
n_best = 3

#-----------------------------------------------------


choose_one_traj = 2
plot_all = False

class TrajProcessing(object):
    def __init__(self):
        self.true_traj = []
        self.draw_traj = []
        self.number_traj = number_traj
        self.n_draw = 0
        self.intrapolated_traj = []

        self.choose_traj = choose_one_traj

        self.true_traj_m = []
        self.draw_traj_m = []

        self.draw_traj_mouse = []
        self.draw_traj_mouse_m = []

        self.n_draw_mouse = 0

        self.erros = []
        self.erros_mouse = []

        self.index_best_draw = 0
        self.index_best_mouse = 0

        self.data_participant = []


    def new_trajectory(self):
        self.true_traj = []
        self.draw_traj = []
        self.number_traj = number_traj
        self.n_draw = 0
        self.intrapolated_traj = []

        self.choose_traj = choose_one_traj

        self.true_traj_m = []
        self.draw_traj_m = []

        self.draw_traj_mouse = []
        self.draw_traj_mouse_m = []

        self.n_draw_mouse = 0

        self.erros = []
        self.erros_mouse = []

        self.index_best_draw = 0
        self.index_best_mouse = 0


    def read_traj_dtime(self, n):
        with open('trajectories/' + str(n_particapant) + '_draw_traj_' + str(n) + '_dtime.csv',
                  newline='') as file:
            data = csv.reader(file)
            print('delta time')
            dtime = []
            for row in data:
                row = np.array(row)
                dtime.append(row[i])
            print(dtime)

    def read_traj_csv_true(self, n):
        with open('trajectories/true_traj_' + str(n) + '.csv', newline='') as file:
            true_traj = []
            data = csv.reader(file)
            for row in data:
                row = np.array(row)
                for i in range(len(row)):
                    true_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", true_traj[i])
                    true_traj[i] = (float(data_float[0]), float(data_float[1]))
                self.true_traj = true_traj

    def read_traj_csv_drawing_all(self, n):
        with open('trajectories/' + str(n_particapant) + '_draw_traj_' + str(n) + '.csv',
                  newline='') as file:
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

    def read_traj_csv_drawing_mouse_all(self, n):
        with open('trajectories/' + str(n_particapant) + '_mouse_traj_' + str(n) + '.csv',
                  newline='') as file:
            data = csv.reader(file)
            self.n_draw_mouse = 0
            for row in data:
                self.n_draw_mouse += 1
                # print(self.n_draw)
                row = np.array(row)
                draw_traj = []
                for i in range(len(row)):
                    draw_traj.append(row[i][1:-1])
                    data_float = re.findall(r"[-+]?\d*\.\d+|\d+", draw_traj[i])
                    draw_traj[i] = (float(data_float[0]), float(data_float[1]))
                self.draw_traj_mouse.append(draw_traj)
            # for i in range(len(self.draw_traj_mouse)):
            #     print(self.draw_traj_mouse[i])

    def plot_one_traj(self, true_traj, draw_traj):
        plt.figure()
        plt.title('Trajectories')
        x_t = []
        y_t = []
        for i in range(len(true_traj)):
            x_t.append(true_traj[i][0])
            y_t.append(true_traj[i][1])
        plt.plot(x_t, y_t, label='True trajectory')
        plt.plot(x_t, y_t, 'bo', markersize=2)
        x = []
        y = []
        for i in range(len(draw_traj)):
            x.append(draw_traj[i][0])
            y.append(draw_traj[i][1])
        plt.plot(x, y, label='True trajectory')
        plt.plot(x, y, 'go', markersize=2)
        plt.legend()
        plt.show()

    def plot_all(self, true_traj, draw_traj, color):
        plt.figure()
        plt.title('Trajectories ALL')
        x_t = []
        y_t = []
        for i in range(len(true_traj)):
            x_t.append(true_traj[i][0])
            y_t.append(true_traj[i][1])
        plt.plot(x_t, y_t, label='True trajectory')
        # plt.plot(x_t, y_t, 'bo', markersize=0)
        for k in range(len(draw_traj)):
            x = []
            y = []
            for i in range(len(draw_traj[k])):
                x.append(draw_traj[k][i][0])
                y.append(draw_traj[k][i][1])
            plt.plot(x, y, color=color, linewidth=1.0, linestyle='--', label='Drawing traj' + str(k))
        plt.legend()
        plt.show()

    def plot_all_drawn_mouse(self, true_traj, draw_traj, color_drawn, mouse_traj, color_mouse):
        plt.figure()
        plt.title('Trajectories ALL, Drawn, Mouse')
        x_t = []
        y_t = []
        for i in range(len(true_traj)):
            x_t.append(true_traj[i][0])
            y_t.append(true_traj[i][1])
        plt.plot(x_t, y_t, label='True trajectory')

        for k in range(len(self.index_best_draw[:, 1])):
            x = []
            y = []
            for i in range(len(draw_traj[k])):
                x.append(draw_traj[int(self.index_best_draw[k, 1])][i][0])
                y.append(draw_traj[int(self.index_best_draw[k, 1])][i][1])
            plt.plot(x, y, color=color_drawn, linewidth=1.0, linestyle='--', label='Drawing traj' + str(k))

        # for k in range(len(draw_traj)):
        #     x = []
        #     y = []
        #     for i in range(len(draw_traj[k])):
        #         x.append(draw_traj[k][i][0])
        #         y.append(draw_traj[k][i][1])
        #     plt.plot(x, y, color=color_drawn, linewidth=1.0, linestyle='--',  label='Drawing traj' + str(k))

        for k in range(len(self.index_best_mouse[:, 1])):
            x = []
            y = []
            for i in range(len(mouse_traj[k])):
                x.append(mouse_traj[int(self.index_best_mouse[k, 1])][i][0])
                y.append(mouse_traj[int(self.index_best_mouse[k, 1])][i][1])
            plt.plot(x, y, color=color_mouse, linewidth=1.0, linestyle='--', label='Mouse traj' + str(k))

        plt.legend()
        plt.show()

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

        if plot_all == True:
            if self.n_draw > 1:
                for k in range(self.n_draw):
                    x = []
                    y = []
                    for i in range(len(self.draw_traj[k])):
                        x.append(self.draw_traj[k][i][0])
                        y.append(self.draw_traj[k][i][1])
                    plt.plot(x, y, label='Drawing traj' + str(k))
        else:
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

        # print(x)
        # print(y)

        self.intrapolated_traj = interpcurve(50, x, y)

    def coordinates_processsing_2d(self):
        frame_size = (640, 480, 50)
        shift = (25, 0.5, 0)
        room_size = (3, 3, 2.5)  # x, y, z

        # transform px to coordinates VICON system (meters)
        def transform(pos, frame_size, shift, room_size):
            new_coordinates = [(pos[0] / frame_size[0] - shift[1]) * room_size[1],
                               ((frame_size[1] - pos[1]) / frame_size[1] + shift[2]) * room_size[2]]
            return new_coordinates

        self.true_traj_m = []
        self.draw_traj_m = []

        # transform true trajectory
        for i in range(len(self.true_traj)):
            # print(self.true_traj[i])
            self.true_traj_m.append(transform(self.true_traj[i], frame_size, shift, room_size))

        # transform drawing trajectories
        if self.n_draw > 1:
            for k in range(self.n_draw):
                x = []
                y = []
                traj_m = []
                for i in range(len(self.draw_traj[k])):
                    x.append(self.draw_traj[k][i][0])
                    y.append(self.draw_traj[k][i][1])
                    traj_m.append(transform((x[i], y[i]), frame_size, shift, room_size))
                self.draw_traj_m.append(traj_m)

        # transform mouse trajectories
        if self.n_draw_mouse > 1:
            for k in range(self.n_draw_mouse):
                x = []
                y = []
                traj_m = []
                for i in range(len(self.draw_traj_mouse[k])):
                    x.append(self.draw_traj_mouse[k][i][0])
                    y.append(self.draw_traj_mouse[k][i][1])
                    traj_m.append(transform((x[i], y[i]), frame_size, shift, room_size))
                self.draw_traj_mouse_m.append(traj_m)

        # # --- ploting ---
        # plt.figure()
        # plt.title('Trajectories_meters')
        # x_t = []
        # y_t = []
        # for i in range(len(self.true_traj_m)):
        #     x_t.append(self.true_traj_m[i][0])
        #     y_t.append(self.true_traj_m[i][1])
        # plt.plot(x_t, y_t, label='True trajectory')
        # plt.plot(x_t, y_t, 'bo', markersize=2)
        #
        # if self.n_draw > 1:
        #     for k in range(self.n_draw):
        #         x = []
        #         y = []
        #         for i in range(len(self.draw_traj_m[k])):
        #             x.append(self.draw_traj_m[k][i][0])
        #             y.append(self.draw_traj_m[k][i][1])
        #         plt.plot(x, y, label='Drawing traj meters' + str(k))
        #
        # plt.legend()
        # plt.show()

    def get_errors(self):
        def get_error(true_trajectory, draw_trajectory):
            true_pose = np.linalg.norm(true_trajectory, axis=1)
            draw_pose = np.linalg.norm(draw_trajectory, axis=1)
            pose_difference = abs(true_pose - draw_pose)
            max_error = np.max(pose_difference)
            mean_error = np.mean(pose_difference)
            rmse = np.sqrt(np.mean((true_pose - draw_pose) ** 2))
            return max_error, mean_error, rmse

        for i in range(len(self.draw_traj_m)):
            max_error, mean_error, rmse = get_error(self.true_traj_m, self.draw_traj_m[i])
            self.erros.append((max_error, mean_error, rmse))
            # print("HAND. Max error: %.2f" % (max_error) + "m , Mean error: %.2f" % (mean_error) + "m , RMSE: %.2f" % (rmse))

        for i in range(len(self.draw_traj_mouse_m)):
            max_error, mean_error, rmse = get_error(self.true_traj_m, self.draw_traj_mouse_m[i])
            self.erros_mouse.append((max_error, mean_error, rmse))
            # print("MOUSE. Max error: %.2f" % (max_error) + "m , Mean error: %.2f" % (mean_error) + "m , RMSE: %.2f" % (rmse))

        self.erros = np.array(self.erros)
        self.index_best_draw = np.array(sorted(zip(self.erros[:, 1], range(len(self.erros))), reverse=False)[:n_best])

        self.erros_mouse = np.array(self.erros_mouse)
        self.index_best_mouse = np.array(
            sorted(zip(self.erros_mouse[:, 1], range(len(self.erros_mouse))), reverse=False)[:n_best])

        # print(self.erros[:,1])
        # print(self.index_best_draw[:,1])

        errors = np.zeros(shape=(2, 3))
        # print(errors)

        max_errors = []
        mean_erros = []
        RMSE_erros = []

        for i in range(len(self.index_best_draw[:, 1])):
            k = int(self.index_best_draw[i, 1])
            print("HAND. Max error: %.2f" % (self.erros[k, 0]) + "m , Mean error: %.2f" % (
                self.erros[k, 1]) + "m , RMSE: %.2f" % (self.erros[k, 2]))
            max_errors.append(self.erros[k, 0])
            mean_erros.append(self.erros[k, 1])
            RMSE_erros.append(self.erros[k, 2])

        errors[0, 0] = np.mean(max_errors)
        errors[0, 1] = np.mean(mean_erros)
        errors[0, 2] = np.mean(RMSE_erros)

        max_errors = []
        mean_erros = []
        RMSE_erros = []

        for i in range(len(self.index_best_mouse[:, 1])):
            k = int(self.index_best_mouse[i, 1])
            print("MOUSE. Max error: %.2f" % (self.erros_mouse[k, 0]) + "m , Mean error: %.2f" % (
                self.erros_mouse[k, 1]) + "m , RMSE: %.2f" % (self.erros_mouse[k, 2]))
            max_errors.append(self.erros_mouse[k, 0])
            mean_erros.append(self.erros_mouse[k, 1])
            RMSE_erros.append(self.erros_mouse[k, 2])

        errors[1, 0] = np.mean(max_errors)
        errors[1, 1] = np.mean(mean_erros)
        errors[1, 2] = np.mean(RMSE_erros)

        print('-------------------------')
        print('Processing trajectory ' + str(len(self.data_participant) + 1))
        print("HAND MEAN. Max error: %.2f" % (errors[0, 0]) + "m , Mean error: %.2f" % (
            errors[0, 1]) + "m , RMSE: %.2f" % (errors[0, 2]))

        print("MOUSE MEAN. Max error: %.2f" % (errors[1, 0]) + "m , Mean error: %.2f" % (
            errors[1, 1]) + "m , RMSE: %.2f" % (errors[1, 2]))

        print('-------------------------')
        self.data_participant.append(errors)

        # self.plot_one_traj(self.true_traj_m, self.draw_traj_m[0])
        # self.plot_all(self.true_traj_m, self.draw_traj_m, 'red')
        # self.plot_all(self.true_traj_m, self.draw_traj_mouse_m, 'green')



    def plot_all(self):
        self.plot_all_drawn_mouse(self.true_traj_m, self.draw_traj_m, 'red', self.draw_traj_mouse_m, 'green')

    def print_total_data(self):
        print('PARTICIPANT ' + str(n_particapant))
        for i in range(3):
            print('TRAJECTORY ' + str(i + 1) + ' ' + names_traj[i])
            print("HAND MEAN. Max error: %.2f" % (self.data_participant[i][0, 0]) + "m , Mean error: %.2f" % (
                self.data_participant[i][0, 1]) + "m , RMSE: %.2f" % (self.data_participant[i][0, 2]))

            print("MOUSE MEAN. Max error: %.2f" % (self.data_participant[i][1, 0]) + "m , Mean error: %.2f" % (
                self.data_participant[i][1, 1]) + "m , RMSE: %.2f" % (self.data_participant[i][1, 2]))



if __name__ == '__main__':
    try:
        node = TrajProcessing()

        for i in range(1, 4):
            print(i)
            node.new_trajectory()
            node.read_traj_csv_true(i)
            node.read_traj_csv_drawing_all(i)
            node.read_traj_csv_drawing_mouse_all(i)
            # node.read_traj_dtime(i)
            node.coordinates_processsing_2d()
            node.get_errors()
            node.plot_all()

        node.print_total_data()

        # while True:
        #     print('1')
        #     # rospy.spin()

    except rospy.ROSInterruptException:
        pass
