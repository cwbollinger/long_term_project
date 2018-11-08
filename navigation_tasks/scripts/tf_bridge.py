#! /usr/bin/env python

import math
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

def main():
    pub_tf = rospy.Publisher("/mapping/tf", TFMessage, queue_size=100)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    t = TransformStamped()

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():

	try:
            trans1 = tfBuffer.lookup_transform('base_link', 'odom', rospy.Time())
            trans2 = tfBuffer.lookup_transform('base_link', 'base_laser_link', rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        pub_tf.publish(TFMessage([trans1, trans2]))
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf_bridge')
    main()
