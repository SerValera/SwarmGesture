#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1251 -*-

#---choose trajectory--- 1, 2  3
traj_number = 1
n_participant = 8

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import time
from collections import deque

# ---ros init---
from geometry_msgs.msg import PoseStamped
import rospy
from std_msgs.msg import String

from scipy import interpolate

# from PIL import ImageFont, ImageDraw, Image

rospy.init_node('gesture', anonymous=True)
pub1 = rospy.Publisher('/hand_position_right', PoseStamped, queue_size=10)
# pub2 = rospy.Publisher('/gesture_number', PoseStamped, queue_size=10)
pub3 = rospy.Publisher('/letter_drone', String, queue_size=10)

pub_send_traj = rospy.Publisher('/send_traj', String, queue_size=10)

data_out_position = PoseStamped()
data_out_gesture = PoseStamped()
rate = rospy.Rate(10)  # 10hz
# --------------

# ---Anabling to draw on the screen by hand---
import matplotlib.pyplot as plt

draw_line = False

import csv
import math
from math import ceil
import statistics
import time
from collections import deque

from sklearn.metrics import mean_squared_error

running_size = 5
collected_gesture = deque([0])
for i in range(running_size - 1):
    collected_gesture.append(i)

# Load the models built in the previous steps
from keras.models import load_model
from tensorflow import keras

mlp_model = load_model('emnist_mlp_model.h5')
cnn_model = load_model('emnist_cnn_model.h5')

# Letters lookup
letters_eng = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
               11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
               21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# letters_rus = { 1: 'А', 2: 'Б', 3: 'В', 4: 'Г', 5: 'Д', 6: 'Е', 7: 'Ё', 8: 'Ж', 9: 'З', 10: 'И',
# 11: 'Й', 12: 'К', 13: 'Л', 14: 'М', 15: 'Н', 16: 'О', 17: 'П', 18: 'Р', 19: 'С', 20: 'Т',
# 21: 'У', 22: 'Ф', 23: 'Х', 24: 'Ц', 25: 'Ч', 26: 'Ш', 27: 'Щ', 28: 'Ъ', 29: 'Ы', 30: 'Ь', 31: 'Э', 32: 'Ю', 33: 'Я'}

letters_rus = {1: 'А', 2: 'Б', 3: 'В', 4: 'Г', 5: 'А', 6: 'Е', 7: 'Ж', 8: 'З', 9: 'И',
               10: 'Й', 11: 'К', 12: 'Л', 13: 'М', 14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т',
               20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч', 25: 'Ш', 26: 'Щ', 27: 'Ъ', 28: 'Н', 29: 'Ь', 30: 'Э',
               31: 'Ю', 32: 'Я'}

letters_rus_psevdo = {1: 'a', 2: 'b', 3: 'v', 4: 'g', 5: 'a', 6: 'e', 7: 'j', 8: 'z', 9: 'i',
                      10: 'i', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'r', 18: 's', 19: 't',
                      20: 'y', 21: 'f', 22: 'x', 23: 'ch', 24: 'ch', 25: 'sh', 26: 'sh', 27: 'b', 28: 'n', 29: 'b',
                      30: 'ee', 31: 'u', 32: 'ya'}

thickness_draw_line = 3
thickness_record_line = 3




# variebles for get position
img_center_x = 250
img_center_y = 250
x_f = 0
y_f = 0
a_f = 0

import cv2
from src.hand_tracker import HandTracker
import math
import numpy as np
from tensorflow.keras.models import load_model

WINDOW_letter = "Letter Tracking"
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

sign_classifier = load_model('models/model2.h5')
SIGNS = ['one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up', 'thumbs_down', 'close']
SIGNS_dict = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'ok': 6,
    'rock': 7,
    'thumbs_up': 8,
    'thumbs_down': 9,
    'close': 10
}

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)


def plot_trajectories(int_traj, draw_traj, color_int, color_draw):
    plt.figure()
    plt.title('Trajectory processing')

    x = []
    y = []
    for i in range(len(draw_traj)):
        x.append(draw_traj[i][0])
        y.append(draw_traj[i][1])
    plt.plot(x, y, 'r', label='gesture given')
    plt.plot(x, y, 'r.')

    x_t = []
    y_t = []
    for i in range(len(int_traj)):
        x_t.append(int_traj[i][0])
        y_t.append(int_traj[i][1])
    plt.plot(x_t, y_t, 'b', label='generated trajectory')
    plt.plot(x_t, y_t, 'b.')

    plt.xlabel("X, px")
    plt.ylabel("Y, px")
    plt.legend()
    plt.show()

def get_size_pulm(pulm_key_point):
    # calculation size of pulm
    k_pulm = len(pulm_key_point)
    distance_pulm = np.zeros(k_pulm - 1)
    size_pulm = 0
    for i in range(k_pulm - 1):
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[i + 1][0]
        y2 = points[i + 1][1]
        distance_pulm[i] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        size_pulm += distance_pulm[i]
    size_pulm = size_pulm / k_pulm
    # print('size_pulm: ', size_pulm)
    return size_pulm


def get_sum_length(points):
    # calculate length of finger
    k = len(points)
    distance = np.zeros(k - 1)
    sum = 0
    for i in range(k - 1):
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[i + 1][0]
        y2 = points[i + 1][1]
        distance[i] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        sum += distance[i]
    sum = sum
    return sum


def gesture_recognition(points):
    fingers = np.ones(5)

    if points[4, 0] < points[5, 0] + 20:
        fingers[0] = False
    if points[8, 1] > points[5, 1] - 20:
        fingers[1] = False
    if points[12, 1] > points[9, 1] - 10:
        fingers[2] = False
    if points[16, 1] > points[13, 1] - 10:
        fingers[3] = False
    if points[20, 1] > points[17, 1] - 10:
        fingers[4] = False

    dist_big_first = math.sqrt((points[4, 0] - points[8, 0]) ** 2 + (points[4, 1] - points[8, 1]) ** 2)

    gesture = ''
    gesture_number = 0
    if (fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4] == False):
        gesture = 'One'
        gesture_number = 1
    if (fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'Five'
        gesture_number = 5
    if (fingers[0] == False and fingers[1] and fingers[2] and fingers[3] == False and fingers[4] == False):
        gesture = 'Two'
        gesture_number = 2
    if (fingers[0] == False and fingers[1] and fingers[2] and fingers[3] and fingers[4] == False):
        gesture = 'Three'
        gesture_number = 3
    if (fingers[0] == False and fingers[1] and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'Four'
        gesture_number = 4
    if (fingers[0] == False and fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4]):
        gesture = 'Rock'
        gesture_number = 6
    if (fingers[0] == False and fingers[1] == False and fingers[2] == False and fingers[3] == False and fingers[
        4] == False):
        gesture = 'Close'
        gesture_number = 9
    if (dist_big_first < 50 and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'OK'
        gesture_number = 7
    if (points[4, 1] < points[8, 1]):
        gesture = 'Thumbs up'
        gesture_number = 8
    # print(fingers)
    # print(gesture)
    return gesture, gesture_number


def draw_lines():
    # ----for the whole one line recorder----
    if len(lines_recorded) != 0:
        # print(len(lines_recorded))
        for i in range(len(lines_recorded)):
            for j in range(len(lines_recorded[i]) - 1):
                cv2.line(frame, (int(lines_recorded[i][j][0]), int(lines_recorded[i][j][1])),
                         (int(lines_recorded[i][j + 1][0]),
                          int(lines_recorded[i][j + 1][1])), (0, 255, 0), thickness=thickness_draw_line)
    # ----for one line recorder----
    if cord_recorded != []:
        if len(cord_recorded) > 1:
            for i in range(len(cord_recorded) - 1):
                cv2.line(frame, (int(cord_recorded[i][0]), int(cord_recorded[i][1])), (int(cord_recorded[i + 1][0]),
                                                                                       int(cord_recorded[i + 1][1])),
                         (0, 255, 0), thickness=thickness_draw_line)

    if posList != []:
        if len(posList) > 1:
            for i in range(len(posList) - 1):
                cv2.line(frame, (640 - int(posList[i][0]), int(posList[i][1])), (640 - int(posList[i + 1][0]),
                                                                                 int(posList[i + 1][1])),
                         (255, 255, 0), thickness=thickness_draw_line)


def onMouse(event, x, y, flags, param):
    global start_mouse
    global posList
    global start

    if event == cv2.EVENT_LBUTTONDOWN and start_mouse == False:
        start_mouse = True
        d_time = 0
        start = time.time()
        time.clock()
        print(start_mouse)

    if event == cv2.EVENT_LBUTTONUP and start_mouse == True:
        start_mouse = False
        end_time = time.time()
        d_time = end_time - start
        print(start_mouse)
        # ---SAVE TRAJECTORIES---
        # with open('trajectories/' + str(n_participant) + '_mouse_traj_' + str(traj_number) + '.csv', 'a') as f:
        #     traj_mouse_int = interpolate_traj(posList, true_traj)
        #     thewriter = csv.writer(f)
        #     thewriter.writerow(traj_mouse_int)
        # posList = []

        with open('trajectories/mouse_drone.csv', 'w') as f:
            traj_mouse_int = interpolate_traj(posList, true_traj)
            thewriter = csv.writer(f)
            thewriter.writerow(traj_mouse_int)
        posList = []

        rospy.sleep(1)
        pub_send_traj.publish('start_mouse')
        rospy.sleep(1)

        with open('trajectories/' + str(n_participant) + '_mouse_traj_' + str(traj_number) + '_dtime.csv', 'a') as f:
            thewriter = csv.writer(f)
            data = []
            data.append(d_time)
            thewriter.writerow(data)

    if start_mouse == True and event == cv2.EVENT_MOUSEMOVE:
        posList.append((x, y))


def gesture_system_control(gesture_ml):
    # ----moving_average, get smooth data about number of gesture----

    collected_gesture.rotate(1)
    collected_gesture[0] = gesture_ml
    average = sum(collected_gesture) / running_size
    rounding = round(average)
    identical = 1
    for i in range(len(collected_gesture)):
        for j in range(len(collected_gesture)):
            if collected_gesture[i] == collected_gesture[j]:
                identical = identical * 1
            else:
                identical = identical * 0
    # print(collected_gesture, average, identical)

    current_gesture_right = gesture_ml

    if (current_gesture_right != previous_gesture_right) and identical:
        # print(self.collected_gesture, average, rounding, identical)
        print('previous:', previous_gesture_right, ', current:', current_gesture_right)

    previous_gesture_right = current_gesture_right


def predicting():
    image = keras.preprocessing.image
    model = keras.models.load_model('models/cyrillic_model.h5')
    img = image.load_img('res.jpeg', target_size=(278, 278))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    result = int(np.argmax(classes))
    print(classes)
    print('letter detected:', result, letters_rus[result])
    return result


def make_trajecotry(number):
    true_traj = []

    move_up = 15
    # --square--
    if number == 1:
        n_inter = 20
        key_points = ((300, 100-move_up), (500, 100-move_up), (500, 300-move_up), (300, 300-move_up))
        for i in range(len(key_points) - 1):
            x = np.linspace(key_points[i][0], key_points[i + 1][0], num=n_inter, endpoint=True)
            y = np.linspace(key_points[i][1], key_points[i + 1][1], num=n_inter, endpoint=True)
            for j in range(len(x)):
                true_traj.append((x[j], y[j]))
        x = np.linspace(key_points[len(key_points) - 1][0], key_points[0][0], num=n_inter, endpoint=True)
        y = np.linspace(key_points[len(key_points) - 1][1], key_points[0][1], num=n_inter, endpoint=True)
        for j in range(len(x)):
            true_traj.append((x[j], y[j]))

    # --circle--
    if number == 2:
        n_inter = 50
        center = (320, 230-move_up)
        r = 100
        dtheta = 2 * math.pi / n_inter
        theta = 0
        for n in range(n_inter):
            theta += dtheta
            x = r * math.cos(theta) + center[0]
            y = r * math.sin(theta) + center[1]
            true_traj.append((x, y))

    # --square--
    if number == 3:
        n_inter = 15
        key_points = ((310, 120-move_up), (470, 320-move_up), (180, 320-move_up))
        for i in range(len(key_points) - 1):
            x = np.linspace(key_points[i][0], key_points[i + 1][0], num=n_inter, endpoint=True)
            y = np.linspace(key_points[i][1], key_points[i + 1][1], num=n_inter, endpoint=True)
            for j in range(len(x)):
                true_traj.append((x[j], y[j]))
        x = np.linspace(key_points[len(key_points) - 1][0], key_points[0][0], num=n_inter, endpoint=True)
        y = np.linspace(key_points[len(key_points) - 1][1], key_points[0][1], num=n_inter, endpoint=True)
        for j in range(len(x)):
            true_traj.append((x[j], y[j]))

    with open('trajectories/true_traj_' + str(number) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(true_traj)

    return true_traj


def draw_trajectory(true_traj, frame, color_p, radius):
    # ---show trajectory by cv---
    for i in range(len(true_traj)):
        cv2.circle(frame, (int(true_traj[i][0]), int(true_traj[i][1])), radius=radius, color=color_p, thickness=-1)
    return frame


def exp_mean(traj_recorded):
    n = len(traj_recorded)
    # print(np.array(traj_recorded))
    traj_recorded = np.array(traj_recorded)
    exp_mean_x = []
    exp_mean_y = []

    for i in range(n):
        exp_mean_x.append(traj_recorded[i][0])
        exp_mean_y.append(traj_recorded[i][1])

    # print(exp_mean_x, exp_mean_y)

    coeff = 0.7
    for i in range(1, n):
        exp_mean_x[i] = exp_mean_x[i - 1] + coeff * (traj_recorded[i][0] - exp_mean_x[i - 1])
        exp_mean_y[i] = exp_mean_y[i - 1] + coeff * (traj_recorded[i][1] - exp_mean_y[i - 1])
    exp_mean = []

    for j in range(len(exp_mean_x)):
        exp_mean.append((exp_mean_x[j], exp_mean_y[j]))
    return exp_mean


def draw_trajectory_line(traj_recorded_mean, img_trajectory, color_l, thickness):
    for i in range(len(traj_recorded_mean) - 1):
        img_trajectory = cv2.line(img_trajectory, (int(traj_recorded_mean[i][0]), int(traj_recorded_mean[i][1])),
                                  (int(traj_recorded_mean[i + 1][0]),
                                   int(traj_recorded_mean[i + 1][1])), color=color_l,
                                  thickness=thickness)
    return img_trajectory


def compare_traj_same_shape(true_traj, target_traj):
    def get_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    error_dist = []
    for i in range(len(true_traj)):
        dist = get_dist(target_traj[i], true_traj[i])
        error_dist.append(dist)

    print('---- error in pixels ----')
    print('mean error (px):', statistics.mean(error_dist))
    print('median (px):', statistics.median(error_dist))
    print('max (px):', max(error_dist))
    print('min (px):', min(error_dist))


def compare_traj(true_traj, target_traj):
    def get_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    error_dist = []
    for i in range(len(target_traj)):
        min_dist = [0, 1000]
        for j in range((len(true_traj))):
            dist = get_dist(target_traj[i], true_traj[j])
            if dist < min_dist[1]:
                min_dist[1] = dist
                min_dist[0] = j
        error_dist.append(min_dist[1])

    print('---- error in pixels ----')
    print('mean error (px):', statistics.mean(error_dist))
    print('median (px):', statistics.median(error_dist))
    print('max (px):', max(error_dist))
    print('min (px):', min(error_dist))
    return


def interpolate_traj(traj, true_traj):
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

    N = len(true_traj)
    x, y = [], []
    for i in range(len(traj)):
        x.append(traj[i][0])
        y.append(traj[i][1])
    x = np.array(x)
    y = np.array(y)

    traj_int = interpcurve(N, x, y)

    return traj_int


def get_hand_parameter_2(points):
    x1, y1 = points[8][0], points[8][1]
    x2, y2 = points[12][0], points[12][1]
    x_c1, y_c1 = (x1 + x2) / 2, (y1 + y2) / 2
    x1, y1 = points[5][0], points[5][1]
    x2, y2 = points[9][0], points[9][1]
    x_c2, y_c2 = (x1 + x2) / 2, (y1 + y2) / 2
    dy = y_c1 - y_c2
    dx = x_c1 - x_c2
    rads = math.atan2(dy, dx)
    degs = math.degrees(rads)
    if (degs < 0):
        degs += 90
    return degs


time_measure = False
start = 0
start_mouse = False

posList = []
cord_recorded = []
lines_recorded = []
recorded_letters = []
inflight = False
previous_gesture_right = 99

running_size = 7
collected_gesture = deque([0])
for i in range(running_size - 1):
    collected_gesture.append(i)

while True:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = image.shape[0], image.shape[1]

    # ---choose trajectory---
    true_traj = make_trajecotry(traj_number)

    points, _ = detector(image)
    # ---Draw lines---
    if draw_line:
        draw_lines()

    if points is not None:
        # for point in points:
        #     x, y = point
        #     cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        # for connection in connections:
        #     x0, y0 = points[connection[0]]
        #     x1, y1 = points[connection[1]]
        #     cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

        #cv2.circle(frame, (int(points[8][0]), int(points[8][1])), THICKNESS * 2, POINT_COLOR, THICKNESS)

        sign_coords = points.flatten() / float(frame.shape[0]) - 0.5
        sign_class = sign_classifier.predict(np.expand_dims(sign_coords, axis=0))
        sign_text = SIGNS[sign_class.argmax()]
        wrist_x = int(points[0][0])
        wrist_y = int(points[0][1])

        # cv2.putText(frame, sign_text, (wrist_x - 20, wrist_y + 10), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # ----my part----
        pulm_key_point = [points[0], points[5], points[17]]
        pulm_size = get_size_pulm(pulm_key_point)
        gesture, gesture_number = gesture_recognition(points)
        x = points[0, 0]
        y = points[0, 1]

        data_out_position.pose.position.x = x / 100
        data_out_position.pose.position.y = pulm_size / 10
        data_out_position.pose.position.z = 5 - y / 100
        gesture_ml = int(SIGNS_dict[sign_text])
        data_out_position.pose.orientation.w = gesture_ml

        # previous_gesture_right = gesture_system_control(gesture_ml, previous_gesture_right)

        data_out_position.pose.orientation.w = gesture_ml

        pub1.publish(data_out_position)
        cv2.putText(frame, sign_text, (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

        ####
        collected_gesture.rotate(1)
        collected_gesture[0] = gesture_ml
        average = sum(collected_gesture) / running_size
        rounding = round(average)
        identical = 1

        for i in range(len(collected_gesture)):
            for j in range(len(collected_gesture)):
                if collected_gesture[i] == collected_gesture[j]:
                    identical = identical * 1
                else:
                    identical = identical * 0
        # print(collected_gesture, average, identical)

        current_gesture_right = gesture_ml
        if (current_gesture_right != previous_gesture_right) and identical:
            print('previous:', previous_gesture_right, ', current:', current_gesture_right)

            # ---Clean screen---
            if (previous_gesture_right == 5) and (current_gesture_right == 7):
                recorded_letters = []
                cord_recorded = []
                lines_recorded = []

            # ---Clean screen---
            if (previous_gesture_right == 1) and (current_gesture_right == 8):
                draw_line = True
                print('take off. Start drawing')

            if (previous_gesture_right == 5) and (current_gesture_right == 8):
                draw_line = False
                print('land. End drawing')

            previous_gesture_right = current_gesture_right
        ####

        # print(gesture_ml)
        # ---record lines from gesture---
        traj_recorded = []

        if draw_line:
            if gesture_ml == 1:
                cord_recorded.append(points[8])
                time_measure = True

            if gesture_ml == 5:
                if time_measure == False:
                    d_time = 0
                    start = time.time()
                    time.clock()

                if len(cord_recorded) != 0:
                    lines_recorded.append(cord_recorded)
                    cord_recorded = []

            if gesture_ml == 6:
                end_time = time.time()
                d_time = end_time - start
                time_measure = False
                if lines_recorded != []:
                    img_trajectory = np.ones((height, width, 3), np.uint8)  # make the background white
                    img_trajectory_2 = np.ones((height, width, 3), np.uint8)  # make the background white

                    for i in range(len(lines_recorded)):
                        for j in range(len(lines_recorded[i]) - 1):
                            traj_recorded.append((width - int(lines_recorded[i][j][0]), int(lines_recorded[i][j][1])))

                    # ---SHOW TRAJECTORIES---
                    # linear intropolation of the traj
                    traj_recorded_orig = traj_recorded

                    traj_recorded = exp_mean(traj_recorded)
                    traj_recorded_int = interpolate_traj(traj_recorded, true_traj)

                    # smooth data traj
                    # traj_recorded_mean = exp_mean(traj_recorded)
                    # traj_recorded_int_mean = exp_mean(traj_recorded_int)

                    # ---show images---
                    # -draw true traj-
                    # img_trajectory = draw_trajectory_line(true_traj, img_trajectory, (255, 255, 255), 1)
                    # img_trajectory = draw_trajectory_line(true_traj, img_trajectory, (255, 255, 255), 1)

                    # -draw recorded-
                    img_trajectory = draw_trajectory(traj_recorded_orig, img_trajectory, (0, 0, 255), 2)
                    img_trajectory = draw_trajectory_line(traj_recorded_orig, img_trajectory, (0, 0, 255), 1)

                    # -draw recorded mean-
                    #img_trajectory = draw_trajectory(traj_recorded, img_trajectory, (0, 255, 0))
                    #img_trajectory = draw_trajectory_line(traj_recorded, img_trajectory, (255, 255, 0), 1)

                    # -draw smooth recorded-
                    # img_trajectory = draw_trajectory(traj_recorded_mean, img_trajectory, (255, 0, 255), 1)
                    # img_trajectory = draw_trajectory_line(traj_recorded_mean, img_trajectory, (255, 0, 255))

                    # -draw interpolate-
                    img_trajectory = draw_trajectory(traj_recorded_int, img_trajectory, (0, 255, 0), 2)
                    img_trajectory = draw_trajectory_line(traj_recorded_int, img_trajectory, (0, 255, 0), 1)

                    # plot_trajectories(traj_recorded_int, traj_recorded_orig, 'red', 'blue')
                    # draw lines
                    # img_trajectory = draw_trajectory(traj_recorded_int, img_trajectory, (255, 0, 255))

                    # print(true_traj)
                    # print(traj_recorded)

                    # --- END SHOW TRAJECTORIES---

                    # ---INTRAPOLATE TRAJ---
                    # new_traj = my_intapolation(traj_recorded, 100)
                    # img_trajectory = draw_trajectory(new_traj, img_trajectory, (255, 255, 0))

                    cv2.imshow('Draw trajectory', img_trajectory)
                    # cv2.imshow('Trajectory 2', img_trajectory_2)

                    # ---COMPARE TRAJECTORIES---
                    # print('delta time (sec):', d_time)
                    # # compare_traj(true_traj, traj_recorded_mean)
                    # compare_traj_same_shape(true_traj, traj_recorded_int_mean)
                    # # print('number of points:', len(traj_recorded_int_mean), len(true_traj))
                    # print('MSE:', mean_squared_error(true_traj, traj_recorded_int_mean))

                    # ---SAVE TRAJECTORIES---
                    with open('trajectories/' + str(n_participant) +  '_draw_traj_' + str(traj_number) + '.csv', 'a') as f:
                        thewriter = csv.writer(f)
                        thewriter.writerow(traj_recorded_int)

                    with open('trajectories/' + str(n_participant) +  '_draw_traj_' + str(traj_number) + '_dtime.csv', 'a') as f:
                        thewriter = csv.writer(f)
                        data = []
                        data.append(d_time)
                        thewriter.writerow(data)



                    # ---SAVE TRAJECTORY FOR DRONE---
                    with open('trajectories/draw_drone.csv', 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(traj_recorded_int)

                    rospy.sleep(1)
                    pub_send_traj.publish('start_traj')


                cord_recorded = []
                lines_recorded = []
        # ------

    #frame = cv2.circle(frame, (640-int(true_traj[0][0]), int(true_traj[0][1])), radius=10,
    #                    color=(0, 0, 255), thickness=-1)
    frame = draw_trajectory_line(true_traj, frame, (0, 0, 255), 3)
    color = (0, 0, 255)

    frame = cv2.flip(frame, 1)



    cv2.setMouseCallback(WINDOW, onMouse)

    #frameBig = cv2.resize(frame, (1200, 900))
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
