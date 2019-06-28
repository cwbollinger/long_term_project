#! /usr/bin/env python

import math
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, PoseStamped

def main():
    pub_pose = rospy.Publisher("/robot_pose", PoseStamped, queue_size=100)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    pose = PoseStamped()

    rate = rospy.Rate(10.0)
    trans = None
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            print(trans)
            trans.child_frame_id = 'new_map'
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue
            rate.sleep()

        pose.header = trans.header
        pose.pose.position = trans.transform.translation
        pose.pose.orientation = trans.transform.rotation
        pub_pose.publish(pose)
        
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pose_publisher')
    main()
