# ---ros lib---
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

import math
import numpy as np

# ---cv2---
import cv2
from src.hand_tracker import HandTracker
from tensorflow.keras.models import load_model





class GestureRecognition(object):
    def __init__(self):
        self.connections

        self.points1
        self.points2
        self.gesture_ml_r = 0
        self.gesture_ml_l = 0

    def cv_init(self):
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
        sign_classifier = load_model('models/model2.h5')
        SIGNS = ['one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up']
        SIGNS_dict = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'ok': 6, 'rock': 7, 'thumbs_up': 8
        }
        POINT_COLOR = (0, 255, 0)
        CONNECTION_COLOR = (255, 0, 0)
        THICKNESS = 2

        cv2.namedWindow(WINDOW)
        capture = cv2.VideoCapture(1)

        if capture.isOpened():
            hasFrame, frame = capture.read()
        else:
            hasFrame = False

        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]



if __name__ == '__main__':
    try:
        node = GestureRecognition()
        node.cv_init()


        while True:
            node.subscriber_gesture()
            node.subscriber_clock()
            rospy.spin()


    except rospy.ROSInterruptException:
        pass