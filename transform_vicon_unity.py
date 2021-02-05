#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped

name_topic = ['/vicon/cf1/cf1',
              '/vicon/cf1/cf2',
              '/vicon/cf1/cf3',
              '/vicon/cf1/cf4',
              '/vicon/cf1/cf5']

# name_topic = ['test_drone1',
#               'test_drone2',
#               'test_drone3',
#               'test_drone4',
#               'test_drone5']

#---publishers for unity3d---
pub1 = rospy.Publisher('test_drone_unity1', PoseStamped, queue_size=10)
pub2 = rospy.Publisher('test_drone_unity2', PoseStamped, queue_size=10)
pub3 = rospy.Publisher('test_drone_unity3', PoseStamped, queue_size=10)
pub4 = rospy.Publisher('test_drone_unity4', PoseStamped, queue_size=10)
pub5 = rospy.Publisher('test_drone_unity5', PoseStamped, queue_size=10)

data_out1 = PoseStamped()
data_out2 = PoseStamped()
data_out3 = PoseStamped()
data_out4 = PoseStamped()
data_out5 = PoseStamped()

def callback_vicon1(msg):
    data_out1.pose.position.x = msg.transform.translation.x
    data_out1.pose.position.y = msg.transform.translation.y
    data_out1.pose.position.z = msg.transform.translation.z
    pub1.publish(data_out1)

def callback_vicon2(msg):
    data_out2.pose.position.x = msg.transform.translation.x
    data_out2.pose.position.y = msg.transform.translation.y
    data_out2.pose.position.z = msg.transform.translation.z
    pub2.publish(data_out2)

def callback_vicon3(msg):
    data_out3.pose.position.x = msg.transform.translation.x
    data_out3.pose.position.y = msg.transform.translation.y
    data_out3.pose.position.z = msg.transform.translation.z
    pub3.publish(data_out3)

def callback_vicon4(msg):
    data_out4.pose.position.x = msg.transform.translation.x
    data_out4.pose.position.y = msg.transform.translation.y
    data_out4.pose.position.z = msg.transform.translation.z
    pub2.publish(data_out4)

def callback_vicon5(msg):
    data_out5.pose.position.x = msg.transform.translation.x
    data_out5.pose.position.y = msg.transform.translation.y
    data_out5.pose.position.z = msg.transform.translation.z
    pub3.publish(data_out5)

if __name__ == '__main__':
    try:
        while True:
            print('Start')
            rospy.init_node('unitytransform', anonymous=True)
            subscriber1 = rospy.Subscriber(name_topic[0], TransformStamped, callback_vicon1)
            subscriber2 = rospy.Subscriber(name_topic[1], TransformStamped, callback_vicon2)
            subscriber3 = rospy.Subscriber(name_topic[2], TransformStamped, callback_vicon3)
            subscriber4 = rospy.Subscriber(name_topic[3], TransformStamped, callback_vicon4)
            subscriber5 = rospy.Subscriber(name_topic[4], TransformStamped, callback_vicon5)
            rospy.spin()

    except rospy.ROSInterruptException:
        pass



