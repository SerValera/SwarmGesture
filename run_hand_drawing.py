#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- coding: cp1251 -*-

import math
import time
from collections import deque

#---ros init---
from geometry_msgs.msg import PoseStamped
import rospy
from std_msgs.msg import String

from PIL import ImageFont, ImageDraw, Image

rospy.init_node('gesture', anonymous=True)
pub1 = rospy.Publisher('/hand_position_right', PoseStamped, queue_size=10)
#pub2 = rospy.Publisher('/gesture_number', PoseStamped, queue_size=10)
pub3 = rospy.Publisher('/letter_drone', String, queue_size=10)

data_out_position = PoseStamped()
data_out_gesture = PoseStamped()
rate = rospy.Rate(10) # 10hz
#--------------

#---Anabling to draw on the screen by hand---
import matplotlib.pyplot as plt
draw_line = True

import math
import time
from collections import deque

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
letters_eng = { 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j',
11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't',
21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 27: '-'}

# letters_rus = { 1: 'А', 2: 'Б', 3: 'В', 4: 'Г', 5: 'Д', 6: 'Е', 7: 'Ё', 8: 'Ж', 9: 'З', 10: 'И',
# 11: 'Й', 12: 'К', 13: 'Л', 14: 'М', 15: 'Н', 16: 'О', 17: 'П', 18: 'Р', 19: 'С', 20: 'Т',
# 21: 'У', 22: 'Ф', 23: 'Х', 24: 'Ц', 25: 'Ч', 26: 'Ш', 27: 'Щ', 28: 'Ъ', 29: 'Ы', 30: 'Ь', 31: 'Э', 32: 'Ю', 33: 'Я'}

letters_rus = { 1: 'А', 2: 'Б', 3: 'В', 4: 'Г', 5: 'А', 6: 'Е', 7: 'Ж', 8: 'З', 9: 'И',
10: 'Й', 11: 'К', 12: 'Л', 13: 'М', 14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т',
20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч', 25: 'Ш', 26: 'Щ', 27: 'Ъ', 28: 'Н', 29: 'Ь', 30: 'Э', 31: 'Ю', 32: 'Я'}

letters_rus_psevdo = { 1: 'a', 2: 'b', 3: 'v', 4: 'g', 5: 'a', 6: 'e', 7: 'j', 8: 'z', 9: 'i',
10: 'i', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'r', 18: 's', 19: 't',
20: 'y', 21: 'f', 22: 'x', 23: 'ch', 24: 'ch', 25: 'sh', 26: 'sh', 27: 'b', 28: 'n', 29: 'b', 30: 'ee', 31: 'u', 32: 'ya'}

thickness_draw_line = 35
thickness_record_line = 35

#variebles for get position
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

sign_classifier = load_model('models/model1.h5')
SIGNS = ['one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up']
SIGNS_dict = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'ok': 6,
    'rock': 7,
    'thumbs_up': 8
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

def get_size_pulm (pulm_key_point):
    #calculation size of pulm
    k_pulm = len(pulm_key_point)
    distance_pulm = np.zeros(k_pulm-1)
    size_pulm = 0
    for i in range(k_pulm - 1):
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[i + 1][0]
        y2 = points[i + 1][1]
        distance_pulm[i] = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        size_pulm += distance_pulm[i]
    size_pulm = size_pulm/k_pulm
    #print('size_pulm: ', size_pulm)
    return size_pulm
def get_sum_length(points):
    #calculate length of finger
    k = len(points)
    distance = np.zeros(k-1)
    sum = 0
    for i in range(k-1):
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[i+1][0]
        y2 = points[i+1][1]
        distance[i] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        sum += distance[i]
    sum = sum
    return sum
def gesture_recognition(points):
    fingers = np.ones(5)

    if points[4,0] < points[5,0]+20:
        fingers[0] = False
    if points[8,1] > points[5,1]-20:
        fingers[1] = False
    if points[12,1] > points[9,1]-10:
        fingers[2] = False
    if points[16,1] > points[13,1]-10:
        fingers[3] = False
    if points[20,1] > points[17,1]-10:
        fingers[4] = False

    dist_big_first = math.sqrt((points[4,0] - points[8,0]) ** 2 + (points[4,1] - points[8,1]) ** 2)

    gesture = ''
    gesture_number = 0
    if (fingers[1] and fingers[2]==False and fingers[3]==False and fingers[4]==False):
        gesture = 'One'
        gesture_number = 1
    if (fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'Five'
        gesture_number = 5
    if (fingers[0]==False and fingers[1] and fingers[2] and fingers[3]==False and fingers[4]==False):
        gesture = 'Two'
        gesture_number = 2
    if (fingers[0]==False and fingers[1] and fingers[2] and fingers[3] and fingers[4]==False):
        gesture = 'Three'
        gesture_number = 3
    if (fingers[0] == False and fingers[1] and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'Four'
        gesture_number = 4
    if (fingers[0] == False and fingers[1] and fingers[2]== False and fingers[3]== False and fingers[4]):
        gesture = 'Rock'
        gesture_number = 6
    if (fingers[0] == False and fingers[1] == False and fingers[2]== False and fingers[3]== False and fingers[4]== False):
        gesture = 'Close'
        gesture_number = 9
    if (dist_big_first < 50 and fingers[2] and fingers[3] and fingers[4]):
        gesture = 'OK'
        gesture_number = 7
    if (points[4,1] < points[8,1]):
        gesture = 'Thumbs up'
        gesture_number = 8
    #print(fingers)
    #print(gesture)
    return gesture, gesture_number

def draw_lines():
    # ----for the whole one line recorder----
    if len(lines_recorded) != 0:
        # print(len(lines_recorded))
        for i in range(len(lines_recorded)):
            for j in range(len(lines_recorded[i])-1):
                cv2.line(frame, (int(lines_recorded[i][j][0]), int(lines_recorded[i][j][1])), (int(lines_recorded[i][j+1][0]),
                                            int(lines_recorded[i][j+1][1])), (0, 255, 0),  thickness=thickness_draw_line)
    #----for one line recorder----
    if cord_recorded != []:
        if len(cord_recorded) > 1:
            for i in range(len(cord_recorded)-1):
                cv2.line(frame, (int(cord_recorded[i][0]), int(cord_recorded[i][1])), (int(cord_recorded[i+1][0]),
                                            int(cord_recorded[i+1][1])), (0, 255, 0),  thickness=thickness_draw_line)


def letter_detection(img_letter):
    blackboard_gray = cv2.cvtColor(img_letter, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.medianBlur(blackboard_gray, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    # print(blackboard_cnts)
    thresh1 = cv2.flip(thresh1, 1)
    cv2.imwrite('res.jpeg', thresh1)
    result = predicting()

    recorded_letters.append(letters_rus_psevdo[result])
    print(recorded_letters)
    cv2.putText(thresh1, letters_rus_psevdo[result], (40,80), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

    for i in range(len(recorded_letters)):
        cv2.putText(thresh1, recorded_letters[i], (40 + i*25, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    frameBig = cv2.resize(thresh1, (1024, 768))
    cv2.imshow(WINDOW_letter, thresh1)

    rospy.loginfo(letters_rus_psevdo[result])
    pub3.publish(letters_rus_psevdo[result])
    return




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
    #print(collected_gesture, average, identical)

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

    points, _ = detector(image)
    #---Draw lines---
    if draw_line:
        draw_lines()

    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

        sign_coords = points.flatten() / float(frame.shape[0]) - 0.5
        sign_class = sign_classifier.predict(np.expand_dims(sign_coords, axis=0))
        sign_text = SIGNS[sign_class.argmax()]
        wrist_x = int(points[0][0])
        wrist_y = int(points[0][1])

        #cv2.putText(frame, sign_text, (wrist_x - 20, wrist_y + 10), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        #----my part----
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
        cv2.putText(frame, sign_text, (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))

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
        #print(collected_gesture, average, identical)

        current_gesture_right = gesture_ml


        if (current_gesture_right != previous_gesture_right) and identical:
            print('previous:', previous_gesture_right, ', current:', current_gesture_right)

            if (previous_gesture_right == 5) and (current_gesture_right == 7):
                recorded_letters = []
                cord_recorded = []
                lines_recorded = []

            previous_gesture_right = current_gesture_right
        ####



        #print(gesture_ml)
        # ---record lines from gesture---
        if draw_line:
            if gesture_ml == 1:
                cord_recorded.append(points[8])
            if gesture_ml == 5:
                if len(cord_recorded) != 0:
                    lines_recorded.append(cord_recorded)
                    cord_recorded = []

            if gesture_ml == 6:
                if lines_recorded != []:
                    img_letter = np.zeros((height, width, 3), np.uint8)  # make the background white
                    for i in range(len(lines_recorded)):
                        for j in range(len(lines_recorded[i]) - 1):
                            cv2.line(img_letter, (int(lines_recorded[i][j][0]), int(lines_recorded[i][j][1])),
                                     (int(lines_recorded[i][j + 1][0]),
                                      int(lines_recorded[i][j + 1][1])), (255, 255, 255), thickness=thickness_record_line)
                    #cv2.imshow(WINDOW_letter, img_letter)
                    letter_detection(img_letter)
                    # cv2.imwrite('WINDOW_letter', img_letter)
                cord_recorded = []
                lines_recorded = []
        #------




    color = (0, 0, 255)
    frame = cv2.flip(frame, 1)
    frameBig = cv2.resize(frame, (1200, 900))
    cv2.imshow(WINDOW, frameBig)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
