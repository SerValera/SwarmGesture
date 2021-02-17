import csv

# ---ros init---
from geometry_msgs.msg import PoseStamped
import rospy

rospy.init_node('gesture', anonymous=True)
pub1 = rospy.Publisher('/hand_position', PoseStamped, queue_size=10)
# pub2 = rospy.Publisher('/gesture_number', PoseStamped, queue_size=10)

data_out_position = PoseStamped()
data_out_gesture = PoseStamped()
rate = rospy.Rate(10)  # 10hz
# --------------


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

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"

# PALM_MODEL_PATH = "models/pose_detection.tflite"
# LANDMARK_MODEL_PATH = "models/pose_landmark_upper_body.tflite"


ANCHORS_PATH = "models/anchors.csv"

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


i = 0

gestures = ('one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up', 'thumbs_down', 'close')

while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

        print(i)
        i += 1
        with open('dataset/all/gesture_' + gestures[9] + '.csv', 'a') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(points)

    color = (0, 0, 255)
    frame = cv2.flip(frame, 1)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
