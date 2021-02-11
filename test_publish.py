#!/usr/bin/env python
# license removed for brevity

#test publish of coordinates of 5 drones.

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

rospy.init_node('publish_position', anonymous=True)
pub1 = rospy.Publisher('test_drone1', PoseStamped, queue_size=10)
pub2 = rospy.Publisher('test_drone2', PoseStamped, queue_size=10)
pub3 = rospy.Publisher('test_drone3', PoseStamped, queue_size=10)
pub4 = rospy.Publisher('test_drone4', PoseStamped, queue_size=10)
pub5 = rospy.Publisher('test_drone5', PoseStamped, queue_size=10)
#pub_str_test = rospy.Publisher('test_drone_str', String, queue_size=10)

rate = rospy.Rate(4) # 10hz
n_drones = 5

class SendPositionNode(object):
    def __init__(self):
        self.n_drones=n_drones
        self.name_drones=[]
        self.coordinates=[]
        self.data1 = PoseStamped()
        self.data2 = PoseStamped()
        self.data3 = PoseStamped()
        self.data4 = PoseStamped()
        self.data5 = PoseStamped()
        self.message = []
        self.test_message = 'd1,0.0,0.0,1.0,d2,1.0,0.0,1.0,d3,2.0,0.0,1.0,d4,3.0,0.0,1.0,d5,4.0,0.0,1.0'

    def initial_position_str(self):
        i=0
        for drone in range(self.n_drones):
            self.coordinates.append([i, 0, 0])
            self.name_drones.append('drone'+str(i))
            i+=1
        print(self.coordinates)

    def initial_position(self):
        self.data1.pose.position.x = 0
        self.data2.pose.position.x = 1
        self.data3.pose.position.x = 2
        self.data4.pose.position.x = 3
        self.data5.pose.position.x = 4
        self.data1.pose.position.z = 1
        self.data2.pose.position.z = 1
        self.data3.pose.position.z = 1
        self.data4.pose.position.z = 1
        self.data5.pose.position.z = 1

    def update_position_test(self):
        for y in range(10):
            x = 0
            self.data1.pose.position.y = y * 0.1
            self.data2.pose.position.y = y * 0.1
            self.data3.pose.position.y = y * 0.1
            self.data4.pose.position.y = y * 0.1
            self.data5.pose.position.y = y * 0.1
            self.publish_position()
            x = y * 0.1
            y = y * 0.1
            z = y * 0.1

        for y in reversed(range(10)):
            x = 0
            self.data1.pose.position.y = y * 0.1
            self.data2.pose.position.y = y * 0.1
            self.data3.pose.position.y = y * 0.1
            self.data4.pose.position.y = y * 0.1
            self.data5.pose.position.y = y * 0.1
            self.publish_position()

    def publish_position(self):
        pub1.publish(self.data1)
        pub2.publish(self.data2)
        pub3.publish(self.data3)
        pub4.publish(self.data4)
        pub5.publish(self.data5)
        #pub_str_test.publish(str(self.coordinates))
        rate.sleep()

    def update_position_test_str(self):
        for y in np.linspace(0,1,11):
            x = 0
            for drone in range(self.n_drones):
                self.coordinates[drone] = [x, y, 1]
                x+=1
            self.publish_position()
        for y in np.linspace(1,0,11):
            x = 0
            for drone in range(self.n_drones):
                self.coordinates[drone] = [x, y, 1]
                x+=1
            self.publish_position()


if __name__ == '__main__':
    try:
        node = SendPositionNode()
        node.initial_position()
        while True:
            node.update_position_test()
    except rospy.ROSInterruptException:
        pass