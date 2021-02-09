#!/usr/bin/env python
# license removed for brevity
import rospy
import crazyflie
import time
import uav_trajectory

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

pos1 = []
pos2 = []
pos3 = []
pos4 = []
pos5 = []

def cf_init(cf_name):
    cf = crazyflie.Crazyflie(cf_name, '/vicon/' + cf_name + '/' + cf_name)

    cf.setParam("commander/enHighLevel", 1)
    cf.setParam("stabilizer/estimator", 2)  # Use EKF
    cf.setParam("stabilizer/controller", 2)  # Use mellinger controller
    return cf


def callback1(msg):
    pos1 = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z
    print('drone1', [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    cf = cf_init(cf_list[0])
    cf.goTo(goal=[x, y, z], yaw=0.0, duration=1.25)


def callback2(msg):
    #print('drone2', [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z

    cf = cf_init(cf_list[1])
    cf.goTo(goal=[x, y, z], yaw=0.0, duration=1.25)

def callback3(msg):
    #print('drone3', [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    pos3 = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

def callback4(msg):
    #print('drone4', [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    pos4 = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

def callback5(msg):
    #print('drone5', [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    pos5 = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]



if __name__ == '__main__':
    try:
        while True:
            print('Init')
            cf_list = ["cf1", "cf2"]

            for cf_i in cf_list:
                cf = cf_init(cf_i)
                cf.takeoff(targetHeight=1.0, duration=2.0)
                time.sleep(2)

            print('Start')
            rospy.init_node('GetNewPosition', anonymous=True)
            subscriver1 = rospy.Subscriber('/test_drone1', PoseStamped, callback1)
            subscriver2 = rospy.Subscriber('/test_drone2', PoseStamped, callback2)
            subscriver3 = rospy.Subscriber('/test_drone3', PoseStamped, callback3)
            subscriver4 = rospy.Subscriber('/test_drone4', PoseStamped, callback4)
            subscriver5 = rospy.Subscriber('/test_drone5', PoseStamped, callback5)
            rospy.spin()

    except rospy.ROSInterruptException:
        pass



