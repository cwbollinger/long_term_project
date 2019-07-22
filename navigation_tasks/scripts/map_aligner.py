#! /usr/bin/env python

import math
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

def main():
    pub_tf = rospy.Publisher("/tf", TFMessage, queue_size=100)
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    t = TransformStamped()

    rate = rospy.Rate(10.0)
    trans = None
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('map', 'base_link', rospy.Time())
            print(trans)
            trans.child_frame_id = 'new_map'
            break
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass
        rate.sleep()

    while not rospy.is_shutdown():
        trans.header.stamp = rospy.Time().now()
        pub_tf.publish(TFMessage([trans]))
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('tf_bridge')
    main()
