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

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "../../catkin_ws/src/swarm_drones/scripts/models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "../../catkin_ws/src/swarm_drones/scripts/models/hand_landmark.tflite"

ANCHORS_PATH = "../../catkin_ws/src/swarm_drones/scripts/models/anchors.csv"

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

while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)

    #----my part
    pulm_key_point = [points[0], points[5], points[17]]
    finger1 = [points[1], points[2], points[3], points[4]]
    finger2 = [points[5], points[6], points[7], points[8]]
    finger3 = [points[9], points[10], points[11], points[12]]
    finger4 = [points[13], points[14], points[15], points[16]]
    finger5 = [points[17], points[18], points[19], points[20]]

    pulm_size = get_size_pulm (pulm_key_point)
    length1 = get_sum_length(finger1)
    length2 = get_sum_length(finger2)
    length3 = get_sum_length(finger3)
    length4 = get_sum_length(finger4)
    length5 = get_sum_length(finger5)

    gesture, gesture_number = gesture_recognition(points)

    #filter data
    x = points[0, 0]
    y = points[0, 1]

    if points is not None:
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

    color = (0, 0, 255)
    cv2.putText(frame, gesture, (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
