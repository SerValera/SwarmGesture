# ---ros init---
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import rospy
from collections import deque


# buttons position
center_pos = [(100, 100), (100, 200)]
button_rad = [50, 50]
but_status = [False, False]
but_color_status = [False, False]
button_thickness = 1
button_thickness_activ = 6
button_color = [(255, 0, 0), (0, 255, 0)]
#--------------


rospy.init_node('gesture', anonymous=True)
pub_right = rospy.Publisher('/hand_position_right', PoseStamped, queue_size=10)
data_out_position_left = PoseStamped()

pub_left = rospy.Publisher('/hand_position_left', PoseStamped, queue_size=10)
data_out_position_right = PoseStamped()

#pub_boarders = rospy.Publisher('/hands_boarder', String, queue_size=10)
#data_out_boarders = 'No intersect'

#pub_parameters = rospy.Publisher('/hands_parameters', String, queue_size=10)
data_out_parameters = []

pub_commands_robot = rospy.Publisher('/robot_control', String, queue_size=10)

rate = rospy.Rate(15)  # 10hz
parameters_points_1 = [1, 0]
parameters_points_2 = [1, 0]
# --------------



# ---Anabling to draw on the screen by hand---
draw_line = False
two_hands_detection = False

import cv2
from src.hand_tracker import HandTracker
import math
import numpy as np
from tensorflow.keras.models import load_model

# ------------
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"
detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)
# ------------



sign_classifier = load_model('models/model2.h5')
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
THICKNESS = 1

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


#--for gesture contol mode--
running_size = 3
collected_gesture = deque([0])
for i in range(running_size - 1):
    collected_gesture.append(i)
previous_gesture_right = 99
#--------------------

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
        print(len(lines_recorded))
        for i in range(len(lines_recorded)):
            for j in range(len(lines_recorded[i]) - 1):
                cv2.line(frame, (int(lines_recorded[i][j][0]), int(lines_recorded[i][j][1])),
                         (int(lines_recorded[i][j + 1][0]),
                          int(lines_recorded[i][j + 1][1])), (0, 255, 0), thickness=15)
    # ----for one line recorder----
    if cord_recorded != []:
        if len(cord_recorded) > 1:
            for i in range(len(cord_recorded) - 1):
                cv2.line(frame, (int(cord_recorded[i][0]), int(cord_recorded[i][1])), (int(cord_recorded[i + 1][0]),
                                                                                       int(cord_recorded[i + 1][1])),
                         (0, 255, 0), thickness=5)


def get_coordinates_hand(points):
    shift = 40
    start_points = (int(min(points[:, 0])) - shift, int(min(points[:, 1])) - shift)
    end_points = (int(max(points[:, 0])) + shift, int(max(points[:, 1])) + shift)
    rect = [start_points, end_points]
    return rect


def right_or_left(points):
    if points[5, 0] < points[17, 0]:
        hand = 'right'
    else:
        hand = 'left'
    return hand


def publish_hand_left(position, gesture_ml):
    data_out_position_left.pose.position.x = position[0]
    data_out_position_left.pose.position.y = position[1]
    data_out_position_left.pose.position.z = position[2]
    data_out_position_left.pose.orientation.w = gesture_ml
    pub_left.publish(data_out_position_left)


def publish_hand_right(position, gesture_ml):
    data_out_position_right.pose.position.x = position[0]
    data_out_position_right.pose.position.y = position[1]
    data_out_position_right.pose.position.z = position[2]
    data_out_position_right.pose.orientation.w = gesture_ml
    pub_right.publish(data_out_position_right)


def get_positon_hand(points):
    x = points[0, 0]
    y = points[0, 1]
    pulm_key_point = [points[0], points[5], points[17]]
    pulm_size = get_size_pulm(pulm_key_point)
    # position = (x/100, pulm_size / 10, 5 - y / 100)
    position = (x, y, pulm_size)
    #position = (int(x), int(y), int(pulm_size))
    return position


def detect_intersection(rectangular_points_hand1, rectangular_points_hand2):
    intersect = 'No intersect'
    x11 = rectangular_points_hand1[0][0]
    x12 = rectangular_points_hand1[1][0]
    x21 = rectangular_points_hand2[0][0]
    x22 = rectangular_points_hand2[1][0]
    if (x11 < x21 and x21 < x12) or (x11 < x22 and x22 < x12) or (x21 < x11 and x11 < x22) or (x21 < x12 and x12 < x22):
        intersect = 'Intersect'
    # print(intersect)
    return intersect


def get_hand_parameter_1(points):
    x1, y1 = points[8][0], points[8][1]
    x2, y2 = points[20][0], points[20][1]
    l_finger = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x1, y1 = points[5][0], points[5][1]
    x2, y2 = points[17][0], points[17][1]
    l_base = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    l = l_finger / l_base
    return l


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


def draw_buttons(frame):
    for i in range(len(center_pos)):
        ### option2
        if but_color_status[i] == True:
            cv2.circle(frame, (int(center_pos[i][0]), int(center_pos[i][1])), button_rad[i], button_color[0],
                       button_thickness_activ)
        else:
            cv2.circle(frame, (int(center_pos[i][0]), int(center_pos[i][1])), button_rad[i], button_color[1],
                       button_thickness_activ)

        ### option1
        # if but_status[i] == 1:
        #     cv2.circle(frame, (int(center_pos[i][0]), int(center_pos[i][1])), button_rad[i], POINT_COLOR, button_thickness_activ)
        # else:
        #     cv2.circle(frame, (int(center_pos[i][0]), int(center_pos[i][1])), button_rad[i], POINT_COLOR,
        #                button_thickness)





def get_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist

#---for running smothing---
size = 5
collecter_x = deque([0])
collecter_y = deque([0])
collecter_z = deque([0])

for i in range(size - 1):
    collecter_x.append(0)
    collecter_y.append(0)
    collecter_z.append(0)
#------------


cord_recorded = []
lines_recorded = []
while True:
    data_out_parameters = []

    hasFrame, frame = capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    # print(frame.shape)
    frame = cv2.flip(frame, 1)

    points, _ = detector(image)

    # ---Draw lines---
    if draw_line:
        draw_lines()

    hand = ' '
    hand2 = ' '

    if points is not None:
        hand = right_or_left(points)
        if hand == 'left':
            hand2 = 'right'
        if hand == 'right':
            hand2 = 'left'

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

        cv2.putText(frame, sign_text + ":" + hand, (wrist_x - 20, wrist_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)
        gesture_ml = int(SIGNS_dict[sign_text])
        position = get_positon_hand(points)

        # #---average running---
        # collecter_x.rotate(1)
        # collecter_y.rotate(1)
        # collecter_z.rotate(1)
        #
        # collecter_x[0], collecter_y[0], collecter_z[0] = position[0], position[1], position[2]
        # position_average = (sum(collecter_x)/size, sum(collecter_y)/size, sum(collecter_z)/size)
        # #print(position_average)
        #
        # position = position_average
        # position_new = get_positon_hand(points)
        # print(position)

        cv2.circle(frame, (int(position[0]), int(position[1])), THICKNESS * 2, (0, 0, 255), 5)

        if gesture_ml == 5 or gesture_ml == 4:
            param_r_1 = get_hand_parameter_1(points)
            param_r_2 = get_hand_parameter_2(points)
            parameters_points_1[0] = param_r_1
            parameters_points_1[1] = param_r_2

        if gesture_ml == 2:
            param_r_2 = get_hand_parameter_2(points)
            parameters_points_1[1] = param_r_2

        if hand == 'left':
            publish_hand_left(position, gesture_ml)
        if hand == 'right':
            publish_hand_right(position, gesture_ml)

        #print('Distance between fingers: ' + str("{:.2f}".format(parameters_points_1[0])), '. Angle of inclination of the hand: ' + str("{:.1f}".format(parameters_points_1[1])))

        text_dist = 'Distance between fingers: ' + str("{:.2f}".format(parameters_points_1[0]))
        text_angle = 'Inclination of the hand (degrees): ' + str("{:.1f}".format(parameters_points_1[1]))
        text_position = 'Hand coordinates (px): ' + str("{:.2f}".format(position[0])) + '; ' + str("{:.2f}".format(position[1]))
        text_dept = 'Depth ' + str("{:.2f}".format(position[2]))
        text_gesture = 'Gesture: ' + sign_text

        shift = 45
        # cv2.putText(frame, text_dist, (25, 25+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(frame, text_angle, (25, 65+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(frame, text_position, (25, 105+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        # cv2.putText(frame, text_dept, (25, 145+shift), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        gesture_ml = int(SIGNS_dict[sign_text])
        # gesture, gesture_number = gesture_recognition(points)

    if two_hands_detection == False:
        if points is not None:
            if hand == 'left':
                publish_hand_left(position, gesture_ml)
                data_out_parameters.append(parameters_points_1)
                data_out_parameters.append([1, 0])
            if hand == 'right':
                publish_hand_right(position, gesture_ml)
                data_out_parameters.append([1, 0])
                data_out_parameters.append(parameters_points_1)

        # ---record lines from gesture---
        if draw_line:
            if gesture_ml == 1:
                cord_recorded.append(points[8])
            if gesture_ml == 5:
                if len(cord_recorded) != 0:
                    lines_recorded.append(cord_recorded)
                    cord_recorded = []
            if gesture_ml == 6:
                cord_recorded = []
                lines_recorded = []
        # ------

        ####---CONTROL SYSTEM------------
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

            ### OPTION1
            # if (previous_gesture_right == 5) and (current_gesture_right == 1):
            #
            #     for i in range(len(center_pos)):
            #         dist = get_distance(points[8], center_pos[i])
            #         print(dist)
            #         if button_rad[i] > dist:
            #             but_status[i] = not but_status[i]

        ### OPTION 2
        if (current_gesture_right == 1):
            for i in range(len(center_pos)):
                dist = get_distance(points[8], center_pos[i])
                if button_rad[i] > dist:
                    but_color_status[i] = True
                else:
                    but_color_status[i] = False

        if (current_gesture_right == 5):
            for i in range(len(center_pos)):
                but_color_status[i] = False

            previous_gesture_right = current_gesture_right
        ####-------------------------


    draw_buttons(frame)

    frameBig = cv2.resize(frame, (1200, 900))
    # print(data_out_parameters)
    #pub_parameters.publish(str(data_out_parameters))
    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
